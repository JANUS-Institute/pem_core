from numpy.typing import ArrayLike as ArrayLike, NDArray as NDArray
from pathlib import Path

# Re-export amisc things
from amisc import Variable as Variable, Component as Component
import amisc.distribution as distribution
import amisc.transform as transform

PathLike = Path | str

__all__ = ["Variable", "Component", "NDArray", "ArrayLike", "PathLike"]
