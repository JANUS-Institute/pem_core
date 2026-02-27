"""Tests for pem_core.data — data loading and standardization utilities."""
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pem_core.data import (
    UNITS,
    DataEntry,
    DataField,
    _df_to_dataset,
    _split_name_and_unit,
    _standardize_data,
    extract_data_arrays,
    get_field_names,
    interpolate_data_instance,
    load_single_dataset,
    load_multiple_datasets,
    OpVarProps
)


# ── helpers ──────────────────────────────────────────────────────────────────

def scalar_field(value=1.0, err=None):
    """Convenience: wrap a plain float in a 0-D DataField."""
    err_da = xr.DataArray(err) if err is not None else None
    return DataField(val=xr.DataArray(value), err=err_da)


def field_1d(values, coords, err=None):
    """Convenience: 1-D DataField with named coordinate 'x'."""
    da = xr.DataArray(values, coords={"x": coords}, dims=["x"])
    err_da = xr.DataArray(err, coords={"x": coords}, dims=["x"]) if err is not None else None
    return DataField(val=da, err=err_da)


# ── _split_name_and_unit ─────────────────────────────────────────────────────

class TestSplitNameAndUnit:
    def test_basic(self):
        name, unit = _split_name_and_unit("Thrust (mN)")
        assert name == "Thrust"
        assert unit == UNITS.millinewton

    def test_no_unit(self):
        name, unit = _split_name_and_unit("Thrust")
        assert name == "Thrust"
        assert unit is None

    def test_extra_whitespace(self):
        name, unit = _split_name_and_unit("  Thrust  (mN)")
        assert name == "Thrust"
        assert unit == UNITS.millinewton

    def test_compound_name(self):
        name, unit = _split_name_and_unit("Ion current (A)")
        assert name == "Ion current"
        assert unit == UNITS.ampere

    def test_compound_unit(self):
        name, unit = _split_name_and_unit("Velocity (m/s)")
        assert name == "Velocity"
        assert unit == UNITS.meter / UNITS.second

    def test_square_brackets(self):
        name, unit = _split_name_and_unit("Thrust [mN]", bracket_type="[]")
        assert name == "Thrust"
        assert unit == UNITS.millinewton

    def test_curly_brackets(self):
        name, unit = _split_name_and_unit("Thrust {mN}", bracket_type="{}")
        assert name == "Thrust"
        assert unit == UNITS.millinewton

# ── _standardize_data ────────────────────────────────────────────────────────

class TestStandardizeData:
    """Each test creates a fresh DataFrame because _standardize_data mutates in-place."""

    def test_unit_conversion(self):
        df = pd.DataFrame({"Thrust (mN)": [100.0], "Discharge Voltage (V)": [300.0], "Mass Flow Rate (mg/s)": [4.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.newton}},
        )
        assert "thrust" in result.columns
        # 100 mN → 0.1 N
        assert math.isclose(result["thrust"].iloc[0], 0.1)

    def test_no_unit_conversion_needed(self):
        df = pd.DataFrame({"thrust (mN)": [100.0], "discharge voltage (V)": [300.0], "mass flow rate (mg/s)": [4.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        assert math.isclose(result["thrust"].iloc[0], 100.0)
        assert math.isclose(result["discharge voltage"].iloc[0], 300.0)

    def test_case_insensitive_columns(self):
        df = pd.DataFrame({"THRUST (mN)": [100.0], "DISCHARGE VOLTAGE (V)": [300.0], "MASS FLOW RATE (mg/s)": [4.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        assert "thrust" in result.columns
        assert "discharge voltage" in result.columns

    def test_missing_op_var_with_default(self):
        df = pd.DataFrame({"thrust (mN)": [100.0], "discharge voltage (V)": [300.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "background pressure": {"unit": UNITS.pascal, "default": 1e-5},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        assert "background pressure" in result.columns
        assert math.isclose(result["background pressure"].iloc[0], 1e-5)

    def test_missing_op_var_no_default_raises(self):
        df = pd.DataFrame({"thrust (mN)": [100.0], "mass flow rate (mg/s)": [4.0]})
        with pytest.raises(ValueError, match="discharge voltage"):
            _standardize_data(
                df,
                operating_vars={
                    "discharge voltage": {"unit": UNITS.volt},
                    "mass flow rate": {"unit": UNITS.mg / UNITS.s},
                },
                qois={"thrust": {"unit": UNITS.millinewton}},
            )

    def test_absolute_uncertainty_unchanged(self):
        df = pd.DataFrame({
            "thrust (mN)": [100.0],
            "thrust absolute uncertainty": [5.0],
            "discharge voltage (V)": [300.0],
            "mass flow rate (mg/s)": [4.0],
        })
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        assert "thrust uncertainty" in result.columns
        assert math.isclose(result["thrust uncertainty"].iloc[0], 5.0)

    def test_relative_uncertainty_converted_to_absolute(self):
        df = pd.DataFrame({
            "thrust (mN)": [100.0, 200.0],
            "thrust relative uncertainty": [0.05, 0.10],
            "discharge voltage (V)": [300.0, 400.0],
            "mass flow rate (mg/s)": [4.0, 5.0],
        })
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        assert "thrust uncertainty" in result.columns
        assert math.isclose(result["thrust uncertainty"].iloc[0], 5.0)   # 5% of 100
        assert math.isclose(result["thrust uncertainty"].iloc[1], 20.0)  # 10% of 200

    def test_relative_uncertainty_after_unit_conversion(self):
        """Relative → absolute conversion uses the already unit-converted value."""
        df = pd.DataFrame({
            "thrust (mN)": [100.0],          # 100 mN → 0.1 N
            "thrust relative uncertainty": [0.10],  # 10%
            "discharge voltage (V)": [300.0],
            "mass flow rate (mg/s)": [4.0],
        })
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.newton}},
        )
        # thrust = 0.1 N, 10% relative → uncertainty = 0.01 N
        assert math.isclose(result["thrust uncertainty"].iloc[0], 0.01)

    def test_rename_map(self):
        df = pd.DataFrame({"Vd (V)": [300.0], "mdot (mg/s)": [4.0], "T (mN)": [10.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
            rename_map={"vd": "discharge voltage", "mdot": "mass flow rate", "t": "thrust"},
        )
        assert "discharge voltage" in result.columns
        assert "mass flow rate" in result.columns
        assert "thrust" in result.columns
        assert math.isclose(result["thrust"].iloc[0], 10.0)

    def test_flow_rate_from_total_and_ratio(self):
        total_flow, ratio = 5.0, 0.9
        expected = total_flow * ratio / (1 + ratio)
        df = pd.DataFrame({
            "total flow rate (mg/s)": [total_flow],
            "anode-cathode flow ratio": [ratio],
            "discharge voltage (V)": [300.0],
        })
        result = _standardize_data(
            df,
            operating_vars={
                "anode mass flow rate": {"unit": UNITS.mg / UNITS.s},
                "discharge voltage": {"unit": UNITS.volt},
            },
            qois={},
        )
        assert "anode mass flow rate" in result.columns
        assert math.isclose(result["anode mass flow rate"].iloc[0], expected)

    def test_flow_rate_missing_columns_raises(self):
        df = pd.DataFrame({"discharge voltage (V)": [300.0]})
        with pytest.raises(ValueError, match="flow rate"):
            _standardize_data(
                df,
                operating_vars={
                    "anode mass flow rate": {"unit": UNITS.mg / UNITS.s},
                    "discharge voltage": {"unit": UNITS.volt},
                },
                qois={},
            )

    def test_all_columns_are_float(self):
        df = pd.DataFrame({"thrust (mN)": [100], "discharge voltage (V)": [300], "mass flow rate (mg/s)": [4]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
        )
        for col in result.columns:
            assert result[col].dtype == float, f"Column '{col}' is not float dtype"

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({"Thrust (mN)": [100.0], "Discharge Voltage (V)": [300.0], "Mass Flow Rate (mg/s)": [4.0]})
        original_columns = list(df.columns)
        original_values = df["Thrust (mN)"].tolist()
        _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.newton}},
        )
        assert list(df.columns) == original_columns
        assert df["Thrust (mN)"].tolist() == original_values

    def test_rename_map_keys_are_case_insensitive(self):
        """rename_map should match regardless of key capitalisation."""
        df = pd.DataFrame({"Vd (V)": [300.0], "Mdot (mg/s)": [4.0], "T (mN)": [10.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": UNITS.volt},
                "mass flow rate": {"unit": UNITS.mg / UNITS.s},
            },
            qois={"thrust": {"unit": UNITS.millinewton}},
            # Mixed-case keys — previously these were silently ignored
            rename_map={"VD": "discharge voltage", "MDOT": "mass flow rate", "T": "thrust"},
        )
        assert "discharge voltage" in result.columns
        assert "thrust" in result.columns


# ── _df_to_dataset ───────────────────────────────────────────────────────────

class TestDfToDataset:
    # shared operating_vars used across most tests
    OP_VARS: dict[str, OpVarProps] = {
        "discharge voltage": {},
        "mass flow rate": {},
    }

    def test_scalar_qoi_is_zero_d(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0, 400.0],
            "mass flow rate": [4.0, 5.0],
            "thrust": [10.0, 15.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})
        assert len(entries) == 2
        for entry in entries:
            assert entry.data["thrust"].val.ndim == 0

    def test_groups_by_operating_condition(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0, 300.0, 400.0, 400.0],
            "mass flow rate": [4.0, 4.0, 5.0, 5.0],
            "thrust": [10.0, 10.0, 15.0, 15.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})
        assert len(entries) == 2
        voltages = {e.operating_condition["discharge voltage"] for e in entries}
        assert voltages == {300.0, 400.0}

    def test_operating_condition_dict_populated(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0],
            "mass flow rate": [4.0],
            "thrust": [10.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})
        oc = entries[0].operating_condition
        assert math.isclose(oc["discharge voltage"], 300.0)
        assert math.isclose(oc["mass flow rate"], 4.0)

    def test_field_qoi_has_coord_dim(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0] * 3,
            "mass flow rate": [4.0] * 3,
            "axial position": [0.0, 0.01, 0.02],
            "ion velocity": [500.0, 600.0, 700.0],
        })
        entries = _df_to_dataset(
            df, self.OP_VARS,
            qois={"ion velocity": {"coords": ("axial position",)}},
        )
        assert len(entries) == 1
        field = entries[0].data["ion velocity"]
        assert "axial position" in field.val.dims
        assert len(field.val) == 3

    def test_uncertainty_stored_in_err(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0, 400.0],
            "mass flow rate": [4.0, 5.0],
            "thrust": [10.0, 15.0],
            "thrust uncertainty": [0.5, 0.7],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})
        for entry in entries:
            assert entry.data["thrust"].err is not None

    def test_no_uncertainty_column_gives_none_err(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0],
            "mass flow rate": [4.0],
            "thrust": [10.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})
        assert entries[0].data["thrust"].err is None

    def test_missing_qoi_column_skipped(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0],
            "mass flow rate": [4.0],
            "thrust": [10.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}, "ion current": {}})
        assert "thrust" in entries[0].data
        assert "ion current" not in entries[0].data

    def test_unit_string_stored_on_field(self):
        df = pd.DataFrame({
            "discharge voltage": [300.0],
            "mass flow rate": [4.0],
            "thrust": [10.0],
        })
        entries = _df_to_dataset(df, self.OP_VARS, qois={"thrust": {"unit": UNITS.millinewton}})
        assert entries[0].data["thrust"].unit == "mN"

    def test_single_operating_variable(self):
        """A single operating variable must not crash (pandas returns a scalar key, not a tuple)."""
        df = pd.DataFrame({
            "discharge voltage": [300.0, 400.0],
            "thrust": [10.0, 15.0],
        })
        entries = _df_to_dataset(df, {"discharge voltage": {}}, qois={"thrust": {}})
        assert len(entries) == 2
        voltages = {e.operating_condition["discharge voltage"] for e in entries}
        assert voltages == {300.0, 400.0}

    def test_duplicate_scalar_value_raises(self):
        """Two different thrust values under the same operating condition is a data error."""
        df = pd.DataFrame({
            "discharge voltage": [300.0, 300.0],
            "mass flow rate": [4.0, 4.0],
            "thrust": [10.0, 99.0],  # contradictory values at same condition
        })
        with pytest.raises(ValueError, match="thrust"):
            _df_to_dataset(df, self.OP_VARS, qois={"thrust": {}})


# ── load_single_dataset ──────────────────────────────────────────────────────

class TestLoadSingleDataset:
    OP_VARS = {
        "discharge voltage": {"unit": UNITS.volt},
        "mass flow rate": {"unit": UNITS.mg / UNITS.s},
    }

    def _write_csv(self, tmp_path, content, name="data.csv"):
        f = tmp_path / name
        f.write_text(content)
        return f

    def test_returns_one_entry_per_operating_condition(self, tmp_path):
        f = self._write_csv(tmp_path,
            "Discharge Voltage (V),Mass Flow Rate (mg/s),Thrust (mN)\n"
            "300.0,4.0,10.0\n"
            "400.0,5.0,15.0\n"
        )
        entries = load_single_dataset(f, self.OP_VARS, qois={"thrust": {"unit": UNITS.millinewton}})
        assert len(entries) == 2

    def test_unit_conversion(self, tmp_path):
        f = self._write_csv(tmp_path,
            "Discharge Voltage (V),Mass Flow Rate (mg/s),Thrust (mN)\n"
            "300.0,4.0,100.0\n"
        )
        entries = load_single_dataset(f, self.OP_VARS, qois={"thrust": {"unit": UNITS.newton}})
        val = float(entries[0].data["thrust"].val.values)
        assert math.isclose(val, 0.1)

    def test_case_insensitive_columns(self, tmp_path):
        f = self._write_csv(tmp_path,
            "DISCHARGE VOLTAGE (V),MASS FLOW RATE (mg/s),THRUST (mN)\n"
            "300.0,4.0,10.0\n"
        )
        entries = load_single_dataset(f, self.OP_VARS, qois={"thrust": {"unit": UNITS.millinewton}})
        assert len(entries) == 1
        assert "thrust" in entries[0].data

    def test_rename_map(self, tmp_path):
        f = self._write_csv(tmp_path,
            "Vd (V),mdot (mg/s),T (mN)\n"
            "300.0,4.0,10.0\n"
        )
        entries = load_single_dataset(
            f, self.OP_VARS,
            qois={"thrust": {"unit": UNITS.millinewton}},
            rename_map={"vd": "discharge voltage", "mdot": "mass flow rate", "t": "thrust"},
        )
        assert len(entries) == 1
        assert "thrust" in entries[0].data

    def test_square_bracket_units(self, tmp_path):
        f = self._write_csv(tmp_path,
            "Discharge Voltage [V],Mass Flow Rate [mg/s],Thrust [mN]\n"
            "300.0,4.0,10.0\n"
        )
        entries = load_single_dataset(
            f, self.OP_VARS,
            qois={"thrust": {"unit": UNITS.millinewton}},
            unit_bracket_type="[]",
        )
        assert len(entries) == 1

    def test_load_multiple_datasets(self, tmp_path):
        f1 = self._write_csv(tmp_path,
            "Discharge Voltage (V),Mass Flow Rate (mg/s),Thrust (mN)\n300.0,4.0,10.0\n",
            name="a.csv",
        )
        f2 = self._write_csv(tmp_path,
            "Discharge Voltage (V),Mass Flow Rate (mg/s),Thrust (mN)\n400.0,5.0,15.0\n",
            name="b.csv",
        )
        entries = load_multiple_datasets([f1, f2], self.OP_VARS, qois={"thrust": {"unit": UNITS.millinewton}})
        assert len(entries) == 2


# ── get_field_names ──────────────────────────────────────────────────────────

class TestGetFieldNames:
    def test_from_data_entries(self):
        entries = [
            DataEntry({"v": 300.0, "m": 4.0}, {"thrust": scalar_field(), "efficiency": scalar_field()}),
            DataEntry({"v": 400.0, "m": 5.0}, {"thrust": scalar_field(), "ion current": scalar_field()}),
        ]
        assert get_field_names(entries) == {"thrust", "efficiency", "ion current"}

    def test_from_data_instances(self):
        instances = [
            {"thrust": scalar_field(), "efficiency": scalar_field()},
            {"thrust": scalar_field(), "ion current": scalar_field()},
        ]
        assert get_field_names(instances) == {"thrust", "efficiency", "ion current"}

    def test_empty(self):
        assert get_field_names([]) == set()

    def test_single_entry(self):
        entries = [DataEntry({"v": 300.0, "m": 4.0}, {"thrust": scalar_field()})]
        assert get_field_names(entries) == {"thrust"}


# ── extract_data_arrays ──────────────────────────────────────────────────────

class TestExtractDataArrays:
    def test_values_and_errors(self):
        entries = [
            DataEntry({"v": 300.0, "m": 4.0}, {"thrust": scalar_field(10.0, err=0.5)}),
            DataEntry({"v": 400.0, "m": 5.0}, {"thrust": scalar_field(15.0, err=0.7)}),
        ]
        result = extract_data_arrays(entries)
        vals, errs = result["thrust"]
        assert set(vals.tolist()) == {10.0, 15.0}
        assert set(errs.tolist()) == {0.5, 0.7}

    def test_missing_error_filled_with_nan(self):
        entries = [DataEntry({"v": 300.0, "m": 4.0}, {"thrust": scalar_field(10.0)})]
        result = extract_data_arrays(entries)
        _, errs = result["thrust"]
        assert np.isnan(errs[0])

    def test_multiple_entries_concatenated(self):
        entries = [
            DataEntry({"v": v, "m": m}, {"thrust": scalar_field(t)})
            for v, m, t in [(300.0, 4.0, 10.0), (400.0, 5.0, 15.0), (500.0, 6.0, 20.0)]
        ]
        result = extract_data_arrays(entries)
        vals, _ = result["thrust"]
        assert len(vals) == 3

    def test_field_qoi_flattened_to_1d(self):
        entries = [
            DataEntry({"v": 300.0, "m": 4.0}, {"velocity": field_1d([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])}),
        ]
        result = extract_data_arrays(entries)
        vals, errs = result["velocity"]
        assert vals.shape == (3,)
        assert np.all(np.isnan(errs))

    def test_field_absent_in_some_entries(self):
        entries = [
            DataEntry({"v": 300.0, "m": 4.0}, {"thrust": scalar_field(10.0)}),
            DataEntry({"v": 400.0, "m": 5.0}, {"thrust": scalar_field(15.0), "ion current": scalar_field(2.5)}),
        ]
        result = extract_data_arrays(entries)
        assert len(result["thrust"][0]) == 2
        assert len(result["ion current"][0]) == 1

    def test_from_data_instances(self):
        instances = [
            {"thrust": scalar_field(10.0, err=0.5)},
            {"thrust": scalar_field(15.0, err=0.7)},
        ]
        result = extract_data_arrays(instances)
        vals, _ = result["thrust"]
        assert len(vals) == 2

    def test_output_keys_are_sorted(self):
        """Key order must be deterministic (alphabetical) regardless of insertion order."""
        entries = [
            DataEntry({"v": 300.0, "m": 4.0}, {
                "thrust": scalar_field(10.0),
                "efficiency": scalar_field(0.5),
                "ion current": scalar_field(2.0),
            }),
        ]
        result = extract_data_arrays(entries)
        assert list(result.keys()) == sorted(result.keys())


# ── interpolate_data_instance ─────────────────────────────────────────────────

class TestInterpolateDataInstance:
    def test_only_shared_fields_returned(self):
        d1 = {
            "velocity": field_1d([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]),
            "density":  field_1d([10.0, 20.0, 30.0], [0.0, 1.0, 2.0]),
        }
        d2 = {"velocity": field_1d([0.0] * 2, [0.5, 1.5])}
        result = interpolate_data_instance(d1, d2)
        assert "velocity" in result
        assert "density" not in result

    def test_result_has_no_uncertainty(self):
        d1 = {"velocity": field_1d([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], err=[0.1, 0.1, 0.1])}
        d2 = {"velocity": field_1d([0.0] * 2, [0.5, 1.5])}
        result = interpolate_data_instance(d1, d2)
        assert result["velocity"].err is None

    def test_linear_interpolation(self):
        """y = x is linear, so interpolation should be exact."""
        d1 = {"velocity": field_1d([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])}
        x_obs = [0.5, 1.5, 2.5]
        d2 = {"velocity": field_1d([0.0] * 3, x_obs)}
        result = interpolate_data_instance(d1, d2)
        assert np.allclose(result["velocity"].val.values, x_obs)

    def test_unit_inherited_from_d2(self):
        d1 = {"velocity": DataField(val=xr.DataArray([1.0, 2.0], coords={"x": [0.0, 1.0]}, dims=["x"]), unit="m/s")}
        d2 = {"velocity": DataField(val=xr.DataArray([1.5], coords={"x": [0.5]}, dims=["x"]), unit="m/s")}
        result = interpolate_data_instance(d1, d2)
        assert result["velocity"].unit == "m/s"

    def test_empty_d1_returns_empty(self):
        d1 = {}
        d2 = {"velocity": field_1d([1.0, 2.0], [0.0, 1.0])}
        assert interpolate_data_instance(d1, d2) == {}

    def test_scalar_field_passthrough(self):
        """0-D fields have no dims to interpolate; the simulation value is returned as-is."""
        d1 = {"thrust": scalar_field(10.0)}
        d2 = {"thrust": scalar_field(10.5)}
        result = interpolate_data_instance(d1, d2)
        assert "thrust" in result
        assert float(result["thrust"].val.values) == pytest.approx(10.0)


# ── string unit support ───────────────────────────────────────────────────────

class TestStringUnits:
    """Verify that plain strings are accepted wherever pint.Unit objects are."""

    def test_standardize_data_string_unit_conversion(self):
        """String units in qois and operating_vars trigger the same conversion as pint.Unit."""
        df = pd.DataFrame({"Thrust (mN)": [100.0], "Discharge Voltage (V)": [300.0], "Mass Flow Rate (mg/s)": [4.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": "V"},
                "mass flow rate": {"unit": "mg/s"},
            },
            qois={"thrust": {"unit": "N"}},  # mN → N
        )
        assert math.isclose(result["thrust"].iloc[0], 0.1)

    def test_standardize_data_string_unit_no_pint_import_needed(self):
        """A user who never imports pint can still get unit conversion."""
        df = pd.DataFrame({"T (mN)": [50.0], "Vd (V)": [200.0]})
        result = _standardize_data(
            df,
            operating_vars={"vd": {"unit": "V"}},
            qois={"t": {"unit": "mN"}},
            rename_map={"t": "t", "vd": "vd"},
        )
        assert math.isclose(result["t"].iloc[0], 50.0)

    def test_df_to_dataset_unit_string_stored_on_field(self):
        """String unit passed via qois is formatted and stored on DataField.unit."""
        df = pd.DataFrame({
            "discharge voltage": [300.0],
            "mass flow rate": [4.0],
            "thrust": [10.0],
        })
        entries = _df_to_dataset(
            df,
            {"discharge voltage": {}, "mass flow rate": {}},
            qois={"thrust": {"unit": "mN"}},
        )
        assert entries[0].data["thrust"].unit == "mN"

    def test_load_single_dataset_string_units_end_to_end(self, tmp_path):
        """load_single_dataset works without importing pint at the call site."""
        f = tmp_path / "data.csv"
        f.write_text(
            "Discharge Voltage (V),Mass Flow Rate (mg/s),Thrust (mN)\n"
            "300.0,4.0,100.0\n"
        )
        entries = load_single_dataset(
            f,
            operating_vars={
                "discharge voltage": {"unit": "V"},
                "mass flow rate": {"unit": "mg/s"},
            },
            qois={"thrust": {"unit": "N"}},  # mN → N
        )
        val = float(entries[0].data["thrust"].val.values)
        assert math.isclose(val, 0.1)

    def test_string_unit_with_default_op_var(self):
        """String unit in an op-var that supplies a default value still converts correctly."""
        df = pd.DataFrame({"thrust (mN)": [100.0], "discharge voltage (V)": [300.0]})
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": "V"},
                "background pressure": {"unit": "Pa", "default": 1e-5},
            },
            qois={"thrust": {"unit": "mN"}},
        )
        assert "background pressure" in result.columns
        assert math.isclose(result["background pressure"].iloc[0], 1e-5)

    def test_string_unit_coords(self, tmp_path):
        """String units in the coords dict are parsed correctly."""
        df = pd.DataFrame({
            "discharge voltage": [300.0] * 3,
            "mass flow rate": [4.0] * 3,
            "axial position (cm)": [0.0, 1.0, 2.0],
            "ion velocity": [500.0, 600.0, 700.0],
        })
        result = _standardize_data(
            df,
            operating_vars={
                "discharge voltage": {"unit": "V"},
                "mass flow rate": {"unit": "mg/s"},
            },
            qois={"ion velocity": {"coords": ("axial position",)}},
            coords={"axial position": "m"},  # cm → m
        )
        # 0, 1, 2 cm → 0, 0.01, 0.02 m
        assert math.isclose(result["axial position"].iloc[1], 0.01)
