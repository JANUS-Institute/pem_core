from .types import ArrayLike, PathLike
from amisc import System, Variable, Component
from typing import Literal

class PEM(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_file(config: PathLike, output_dir=None, timestamp_prefix="pem", create_subdirs=False) -> "PEM":
        """
        Load a PEM system from a YAML config file.
        @param config: Path to the YAML config file.
        @param output_dir: Optional path to the directory where output files will be saved. If `None`, then output files will be saved in the same directory as the config file.
        @return: A `PEM` object representing the PEM.
        """
        system = System.load_from_file(config)
        pem = PEM(**system.__dict__)
        pem.timestamp_prefix = timestamp_prefix
        pem.create_subdirs = create_subdirs

        # Set directory in which we'll place output files
        if output_dir is not None:
            pem.root_dir = output_dir

        return pem
        
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