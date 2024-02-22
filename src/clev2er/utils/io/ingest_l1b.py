"""Module for ingesting l1b files.

unpack(): Reads a variable from an l1b file. Extrapolates variables with lower frequencies to the 
        same length as those with higher frequencies.

Author: Ben Palmer
Date: 22 Feburary 2024
"""

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from scipy.interpolate import interp1d


def unpack(variable: str, l1b: Dataset, time_var_1: str, time_var_2: str) -> np.ndarray:
    """Reads a variable from an L1b file and extrapolates to higher frequency if needed.

    Reads a variable from an NetCDF L1B file. If the length of the variable is the same as
    time_var_1, use as read. If the length is the same as time_var_2, extrapolate to the same
    length as as time_var_1.

    Args:
        variable: Name of the variable to read from the file.
        l1b: The netCDF4 Dataset file.
        time_var_1: The time variable from the file with the HIGHER frequency (e.g. time_20_ku)
        time_var_2: The time variable from the file with the LOWER frequency (e.g. time_cor_01)

    Returns:
        The variable read from the file

    Raises:
        KeyError: time_var_1 and time_var_2 are the wrong way round.
    """

    time_var_long = np.ma.filled(l1b.variables[time_var_1], np.nan)
    time_var_short = np.ma.filled(l1b.variables[time_var_2], np.nan)

    if time_var_long.size < time_var_short.size:
        raise KeyError("time_var_1 is shorter than time_var_2. Swap these!")

    var = (l1b.variables[variable][:]).astype(float)

    if var.size == time_var_short.size:
        var = interp1d(time_var_short, var, fill_value="extrapolate")(time_var_long)

    return var
