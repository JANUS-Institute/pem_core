"""
Utilities for loading and standardizing experimental data from CSV files.

Data is organized around the concept of an *operating condition* — a unique
combination of values for the operating variables (e.g. discharge voltage,
mass flow rate). Each operating condition produces one `DataEntry`, which
contains one `DataField` per quantity of interest (QoI). Scalar QoIs are
stored as 0-D xarray DataArrays; spatially-resolved field quantities (e.g.
plasma profiles) are stored as 1-D or 2-D DataArrays indexed by their
coordinate dimensions.

Typical workflow:

    1. Declare ``operating_vars`` and ``qois`` dicts that specify the expected
       column names, target units, and (for fields) coordinate names.
       Optionally specify one or more derived columns to apply data transformations.
   2. Call `load_single_dataset` or `load_multiple_datasets` to read one or
       more CSV files. Column names are matched case-insensitively; units
       embedded in headers (e.g. ``"Thrust (mN)"``) are parsed and converted
       automatically.
    3. Use `interpolate_data_instance` to compare simulation output to
       observations on a common grid, and `extract_data_arrays` to flatten
       the structured data into plain NumPy arrays for use in a likelihood.
"""
import pint
from typing import Callable, Literal, Mapping, cast, TypedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from pem_core.types import PathLike, NDArray

@dataclass
class DataField:
    """A single measured or simulated quantity at one operating condition.

    `val` and `err` are xarray DataArrays so that both scalar quantities (0-D)
    and spatially-resolved field quantities (1-D or 2-D, indexed by coordinate
    dimensions such as axial position) are handled uniformly.
    """
    val: xr.DataArray
    err: xr.DataArray | None = None  # measurement uncertainty; None if not provided
    unit: str | None = None

# Maps QoI name → DataField for a single operating condition
DataInstance = dict[str, DataField]

@dataclass
class DataEntry:
    """Experimental data at a single operating condition.

    `operating_condition` holds the values of all operating variables (e.g.
    discharge voltage, mass flow rate) that define this point, and `data` maps
    each QoI name to its corresponding `DataField`.
    """
    operating_condition: dict[str, float]
    data: DataInstance

# Typed dicts for better hinting
class QoIProps(TypedDict, total=False):
    """Configuration for a quantity of interest (QoI).

    `unit` is the target unit after standardization; accepts either a
    ``pint.Unit`` object or a plain string (e.g. ``"mN"``). `coords` lists
    the names of coordinate columns (e.g. ``("axial position",)``) for field
    quantities; omit for scalar QoIs.
    """
    unit: pint.Unit | str
    coords: tuple[str, ...]

class OpVarProps(TypedDict, total=False):
    """Configuration for an operating variable.

    `unit` is the target unit after standardization; accepts either a
    ``pint.Unit`` object or a plain string (e.g. ``"V"``). `default` is used
    to fill in the column when it is absent from a CSV file.
    """
    unit: pint.Unit | str
    default: float

@dataclass
class DerivedColumn:
    """Specification for a column that can be computed from other columns.

    When `_standardize_data` encounters a missing column whose name matches
    `target`, it checks whether all `required` columns are present. If so it
    calls ``compute(df)`` to produce the new column and records `unit_from` as
    the source for its unit (so that subsequent unit conversion works
    correctly). Specs are tried in list order; the first one whose required
    columns are all available wins.

    Args:
        target: Name of the column to produce (after casefolding / renaming).
        required: Column names that must be present for this transform to fire.
        compute: Function that receives the full DataFrame and returns a Series
            for the new column.
        unit_from: If given, the derived column inherits its unit from this
            source column. If ``None``, no unit is tracked (the column is
            treated as dimensionless / already in target units).
    """
    target: str
    required: list[str]
    compute: Callable[[pd.DataFrame], pd.Series]
    unit_from: str | None = None

# Load pint unit registry and define any custom units we need
UNITS = pint.UnitRegistry()

# The base unit "torr" in pint is lowercase. This gives an uppercase version.
UNITS.define("Torr = 1 torr = Torr")

def _parse_unit(u: pint.Unit | str | None) -> pint.Unit | None:
    """Coerce a unit string or pint.Unit to a pint.Unit, or return None."""
    if u is None or isinstance(u, pint.Unit):
        return u
    assert isinstance(u, str)
    return UNITS.parse_units(u)

BracketType = Literal['()', '[]', '{}']

def _split_name_and_unit(key: str, bracket_type: BracketType | None = None):
    """Split a column header like 'Axial ion velocity (m/s)' into name and unit.

    The unit suffix is optional; 'Thrust' returns ('Thrust', None).

    When `bracket_type` is ``None`` (default), the bracket style is
    auto-detected per column by scanning for ``'()'``, ``'[]'``, and ``'{}'``
    groups that contain a valid pint unit. Raises ``ValueError`` if more than
    one unit specification is found in the header, or if content follows the
    unit specification (both indicate likely data-curation errors).

    Pass an explicit `bracket_type` to restrict detection to that style only.
    """
    pairs_to_try: list[BracketType] = [bracket_type] if bracket_type is not None else ['()', '[]', '{}']

    # Collect all bracket groups (across all types to try) that parse as a valid unit.
    matches: list[tuple[int, int, pint.Unit]] = []
    for bt in pairs_to_try:
        left, right = bt
        start = 0
        while True:
            open_ind = key.find(left, start)
            if open_ind == -1:
                break
            close_ind = key.find(right, open_ind + 1)
            if close_ind == -1:
                break
            unit_str = key[open_ind + 1:close_ind].strip()
            if unit_str:
                try:
                    matches.append((open_ind, close_ind, UNITS.parse_units(unit_str)))
                except pint.UndefinedUnitError:
                    pass
            start = close_ind + 1

    if len(matches) > 1:
        raise ValueError(
            f"Multiple unit specifications found in column header '{key}'. "
            "The unit should appear only once, at the end of the header."
        )

    if not matches:
        return key.strip(), None

    open_ind, close_ind, unit = matches[0]

    trailing = key[close_ind + 1:].strip()
    if trailing:
        raise ValueError(
            f"Unexpected content '{trailing}' after unit specification in "
            f"column header '{key}'. The unit must be the last element in the header."
        )

    return key[:open_ind].strip(), unit

def _standardize_data(
        df: pd.DataFrame,
        operating_vars: dict[str, OpVarProps],
        qois: dict[str, QoIProps],
        coords: Mapping[str, pint.Unit | str] | None = None,
        rename_map: dict[str,str] | None = None,
        bracket_type: BracketType | None = None,
        derived_cols: list[DerivedColumn] | None = None,
    ) -> pd.DataFrame:
    """Standardize the dataframe by ensuring required columns are present and appropriately scaled. This includes:
    - Scaling and renaming columns to match expected names and units.
    - Adding any missing operating condition columns with default values so that we have a complete set of operating conditions for each row.
    - Ensuring column names are consistent (e.g. case-insensitive matching, ignoring units in parentheses when matching column names).
    """
    df = df.copy()
    if rename_map is not None:
        rename_map = {k.casefold(): v.casefold() for k, v in rename_map.items()}

    # 1. Strip units from column names and save them in a separate dict so we can convert units later
    #    Also rename columns and casefold for consistent matching.
    unit_dict = {}
    for col in df.columns:
        name, unit = _split_name_and_unit(col, bracket_type=bracket_type)
        name_cf = name.casefold()
        if rename_map is not None and name_cf in rename_map:
            name_cf = rename_map[name_cf]
        unit_dict[name_cf] = unit
        df.rename(columns={col: name_cf}, inplace=True)

    # 1b. Normalize error column names and track their parent QOI
    relative_uncertainty_cols = set()
    error_rename = {}
    for col in list(df.columns):
        for suffix in ("relative uncertainty", "absolute uncertainty"):
            if col.endswith(suffix):
                parent = col[: -len(suffix)].strip()
                if parent in qois:
                    error_rename[col] = f"{parent} uncertainty"

                if suffix == "relative uncertainty":
                    relative_uncertainty_cols.add(f"{parent} uncertainty")
                break

    df.rename(columns=error_rename, inplace=True)
    for old, new in error_rename.items():
        unit_dict[new] = unit_dict.pop(old, None)

    # 2. Apply any caller-supplied derived-column transforms (in order; first match wins per target).
    for derived in (derived_cols or []):
        if derived.target not in df.columns:
            if all(r in df.columns for r in derived.required):
                df[derived.target] = derived.compute(df)
                unit_dict[derived.target] = unit_dict.get(derived.unit_from) if derived.unit_from else None

    # 3. Perform unit conversions as needed
    for col, unit in unit_dict.items():
        if col.endswith(" uncertainty"):
            parent = col[: -len(" uncertainty")].strip()
            if parent in qois:
                expected_unit = _parse_unit(qois[parent].get("unit", UNITS.dimensionless))
            else:
                continue
        elif col in operating_vars:
            expected_unit = _parse_unit(operating_vars[col].get("unit", UNITS.dimensionless))
        elif col in qois:
            expected_unit = _parse_unit(qois[col].get("unit", UNITS.dimensionless))
        elif coords is not None and col in coords:
            expected_unit = _parse_unit(coords[col])
        else:
            continue
        if unit is not None and expected_unit is not None and unit != expected_unit:
            conversion_factor = (1 * unit).to(expected_unit).magnitude
            df[col] = df[col] * conversion_factor
            unit_dict[col] = expected_unit

    # 3b. Convert relative uncertainties to absolute
    for col in df.columns:
        if col.endswith(" uncertainty"):
            parent = col[: -len(" uncertainty")].strip()
            # TODO: do we need this line?
            # Check if this was originally relative (we can track this with a set)
            if col in relative_uncertainty_cols:  # track during rename step above
                df[col] = df[col] * df[parent]

    # 4. Add any missing operating condition columns with default values
    for op_var, props in operating_vars.items():
        if op_var not in df.columns:
            default_value = props.get('default', None)
            if default_value is None:
                raise ValueError(f"Missing required operating variable '{op_var}' and no default value specified in operating_vars.")
            df[op_var] = default_value
            unit_dict[op_var] = _parse_unit(props.get("unit", UNITS.dimensionless))

    # 5. Ensure all columns have float type
    for col in df.columns:
        df[col] = df[col].astype(float)

    return df


def _df_to_dataset(
        df: pd.DataFrame,
        operating_vars: dict[str, OpVarProps],
        qois: dict[str, QoIProps]
    ) -> list[DataEntry]:
    def process_group(group: pd.DataFrame, qois) -> DataInstance:
        data_instance = {}

        for qoi_name, props in qois.items():
            if qoi_name not in group.columns:
                continue

            # Extract coordinates and units for this QoI.
            qoi_coords = list(props.get("coords", ()))
            unit = _parse_unit(props.get("unit", None))
            unit_str = f"{unit:~P}" if unit is not None else ""
            
            # Assemble list of columns we'll extract from the dataframe for this QoI.
            # Start with the coordinates (if present) and the QoI itself.
            # Then look for error columns and associate them with the parent QoI if they exist.
            df_cols = qoi_coords + [qoi_name]

            err_col = f"{qoi_name} uncertainty"
            if err_col in group.columns:
                df_cols.append(err_col)
    
            if len(qoi_coords) == 0:
                # If there are no coordinates, we just want to extract the value as a scalar and wrap it in a DataArray with a dummy dimension so we can keep the same Field structure.
                # Check that there's only one value to flag potential data curation issues.
                df_reduced = group[df_cols].drop_duplicates()
                if len(df_reduced) != 1:
                    raise ValueError(
                        f"Expected only one unique value for QoI '{qoi_name}' "
                        f"but found {len(df_reduced)}. Check the data for consistency."
                    )
                val = xr.DataArray(df_reduced[qoi_name].values[0])
                err = xr.DataArray(df_reduced[err_col].values[0]) if err_col in df_reduced.columns else None
            else:
                # This should work for both 1D and 2D QoIs. xarray will automatically handle the multi-indexing for us when we set the coordinates.
                da = group[df_cols].set_index(qoi_coords).to_xarray()
                val, err = da[qoi_name], da[err_col] if err_col in group.columns else None
            
            field = DataField(val=val, err=err, unit=unit_str)
            
            data_instance[qoi_name] = field

        return data_instance

    data: list[DataEntry] = []
    for opcond, group in df.groupby(list(operating_vars.keys())):
        # pandas returns a scalar (not a 1-tuple) when groupby has a single key
        if not isinstance(opcond, tuple):
            opcond = (opcond,)
        opcond_dict = cast(dict[str, float], dict(zip(operating_vars.keys(), opcond)))
        data_instance = process_group(group, qois)
        data.append(DataEntry(operating_condition=opcond_dict, data=data_instance))
        
    return data

def load_single_dataset(
    file: PathLike,
    operating_vars: dict[str, OpVarProps],
    qois: dict[str, QoIProps],
    coords: Mapping[str, pint.Unit | str] | None = None,
    rename_map: dict[str,str] | None = None,
    unit_bracket_type: BracketType | None = None,
    derived_cols: list[DerivedColumn] | None = None,
) -> list[DataEntry]:
    """Load a CSV file and return one `DataEntry` per unique operating condition.

    Column names are matched case-insensitively. Units embedded in column
    headers (e.g. ``"Thrust (mN)"`` or ``"Thrust [mN]"``) are parsed and
    converted to the target units declared in `qois` / `operating_vars`.
    The bracket style (``'()'``, ``'[]'``, or ``'{}'``) is auto-detected
    per column by default; pass `unit_bracket_type` to restrict to a single
    style. Use `rename_map` to alias column names that differ from the
    expected keys, `coords` to declare the units of any coordinate columns
    used by field QoIs, and `derived_cols` to compute columns that may be
    missing from the CSV.
    """
    df = pd.read_csv(file)
    df_standardized = _standardize_data(df, operating_vars, qois, coords=coords, rename_map=rename_map, bracket_type=unit_bracket_type, derived_cols=derived_cols)
    dataset = _df_to_dataset(df_standardized, operating_vars, qois)
    return dataset

def load_multiple_datasets(
    files: list[PathLike],
    operating_vars: dict[str, OpVarProps],
    qois: dict[str, QoIProps],
    coords: Mapping[str, pint.Unit | str] | None = None,
    rename_map: dict[str,str] | None = None,
    unit_bracket_type: BracketType | None = None,
    derived_cols: list[DerivedColumn] | None = None,
) -> list[DataEntry]:
    """Load and merge multiple CSV files into a single flat list of `DataEntry` objects.

    Each file is processed independently via `load_single_dataset` and the
    results are concatenated. All files must share the same `operating_vars`
    and `qois` schema.
    """
    # TODO: check for and unify duplicate operating conditions
    all_data = []
    for file in files:
        data = load_single_dataset(file, operating_vars, qois, coords=coords, rename_map=rename_map, unit_bracket_type=unit_bracket_type, derived_cols=derived_cols)
        all_data.extend(data)
    return all_data

def interpolate_data_instance(d1: DataInstance, d2: DataInstance) -> DataInstance:
    """Interpolate `d1` onto the coordinate grid of `d2`.

    Useful for comparing simulation output (`d1`) to observations (`d2`) when
    they live on different grids. Only fields present in both instances are
    returned; the result carries no uncertainty (`err=None`).
    """
    itp: DataInstance = {}

    for field_name, data_field in d2.items():
        if field_name in d1:
            da_sim = d1[field_name].val
            da_data = data_field.val
            dims_to_interp = {dim: da_data[dim] for dim in da_data.dims if da_data.sizes[dim] > 1}
        
            da_itp = da_sim.interp(dims_to_interp)
            itp[field_name] = DataField(val=da_itp, err=None, unit=data_field.unit)

    return itp

def get_field_names(arr: list[DataInstance] | list[DataEntry]) -> set[str]:
    """Extract all of the data fields present in a list of DataEntry or DataInstance objects."""
    field_names = set()
    for instance in arr:
        if isinstance(instance, DataEntry):
            field_names.update(instance.data.keys())
        else:
            field_names.update(instance.keys())
    return field_names

def extract_data_arrays(arr: list[DataInstance] | list[DataEntry]) -> dict[str, tuple[NDArray, NDArray]]:
    """Flatten all entries in `arr` into a pair of 1-D arrays per field.

    Returns ``{field_name: (values, errors)}`` where each array concatenates
    values across all operating conditions. Missing errors are filled with NaN.
    """
    field_names = get_field_names(arr)

    output: dict[str, tuple[NDArray, NDArray]] = {}

    for field_name in sorted(field_names):
        vec = []
        err = []

        for entry in arr:
            instance = entry.data if isinstance(entry, DataEntry) else entry
            if field_name in instance.keys():
                field = instance[field_name]
                vec.append(field.val.values.flatten())
                err.append(field.err.values.flatten() if field.err is not None else np.full_like(vec[-1], np.nan))

        output[field_name] = np.concatenate(vec), np.concatenate(err)

    return output