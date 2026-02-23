from .types import ArrayLike, PathLike, Array
from amisc import System, Variable, Component
from typing import Literal
import numpy as np
import os
from pathlib import Path

class PEM(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, config: PathLike, output_dir=None) -> "PEM":
        """
        Load a PEM system from a YAML config file.
        @param config: Path to the YAML config file.
        @param output_dir: Optional path to the directory where output files will be saved. If `None`, then output files will be saved in the same directory as the config file.
        @return: A `PEM` object representing the PEM.
        """
        pem = super().load_from_file(config, root_dir=output_dir, timestamp_prefix="pem")
        pem.__class__ = cls

        return pem
    
    @staticmethod
    def from_directory(directory: PathLike) -> System:
        """
        Given a directory, find the first yaml file in that directory and load the amisc system from that file.
        """
        dir_contents = os.listdir(directory)
        for file_or_dir in dir_contents:
            # Skip directories
            if os.path.isdir(file_or_dir):
                continue

            # Skip files without .yml or .yaml extensions
            file = Path(file_or_dir)
            ext = file.suffix.casefold()
            if ext not in {".yml", ".yaml"}:
                continue

            # Load the system from the file
            return PEM.from_file(directory / file)

        # If we're here, we didn't find anything and should error.
        raise ValueError(f"Could not find a yaml file in directory {directory}.")

        
    def get_nominal_inputs(self, norm=True) -> dict[str, ArrayLike]:
        """Create a dict mapping system inputs to their nominal values"""
        nominals = {}
        for input in self.inputs():
            value = input.get_nominal()
            if norm:
                value = input.normalize(value)
            nominals[input.name] = value

        return nominals
        
    def get_inputs_by_category(self, category, sort: Literal['name', 'tex'] | None = 'name') -> list[Variable]:
        def get_component_variables_by_category(component: Component, category: str, sort: str | None = None) -> list[Variable]:
            """Extract variables of a given `category` from the provided `component`.
            Optionally sort them based on either their 'name' or 'tex' representations"""
            params = [p for p in component.inputs if p.category == category]
            if sort is None:
                return params

            names = [getattr(p, sort).casefold() for p in params]
            sorted_params = [p for _, p in sorted(zip(names, params))]
            return sorted_params
        
        params_by_component = [
            get_component_variables_by_category(component, category, sort=sort) for component in self.components
        ]

        # Flatten the list of lists and remove duplicates while preserving order
        seen = set()
        calibration_vars: list[Variable] = []

        for params in params_by_component:
            for p in params:
                if p not in seen:
                    calibration_vars.append(p)
                    seen.add(p)

        return calibration_vars

    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    


def read_dlm(file: PathLike, delimiter: str | None = ',', comments='#') -> dict[str, Array]:
    """Read a simple delimited file consisting of headers and numerical data into a dict that maps names to columns"""
    with open(file, 'r') as fd:
        header = fd.readline().rstrip()
        if header.startswith(comments):
            header = header[1:].lstrip()

    col_names = header.split(delimiter)
    table_data = np.atleast_2d(np.genfromtxt(file, skip_header=1, delimiter=delimiter))
    columns = [table_data[:, i] for i in range(len(col_names))]
    return {col_name: column for (col_name, column) in zip(col_names, columns)}