"""
Defines all the types used in the module
"""
from pathlib import Path
from typing import Union

import dask
import numpy as np

PathLike = Union[Path, str]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
