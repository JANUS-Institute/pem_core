from numpy.typing import ArrayLike as ArrayLike, NDArray as NDArray
from pathlib import Path
from amisc import Variable as Variable, Component as Component

PathLike = Path | str

__all__ = ["Variable", "Component", "NDArray", "ArrayLike", "PathLike"]
