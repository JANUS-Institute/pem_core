from numpy.typing import ArrayLike as ArrayLike, NDArray as Array
from pathlib import Path
from amisc import Variable as Variable, Component as Component
from amisc.typing import Dataset as Dataset

PathLike = Path | str

__all__ = ["Dataset", "Variable", "Component", "Array", "ArrayLike", "PathLike"]
