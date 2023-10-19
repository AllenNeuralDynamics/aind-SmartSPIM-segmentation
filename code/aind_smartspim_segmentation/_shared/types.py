"""
Defines all the types used in the module
"""
import dask

import numpy as np

from pathlib import Path
from typing import Union

PathLike = Union[Path, str]
ArrayLike = Union[dask.array.core.Array, np.ndarray]