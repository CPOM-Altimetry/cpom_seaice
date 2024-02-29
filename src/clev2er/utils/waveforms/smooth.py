"""Module for smoothing waveforms.

smooth_waveform(): Smooth a waveform using a moving average.


Author: Ben Palmer 
Date: 29 Feb 2024
"""

import numpy as np
import numpy.typing as npt


def smooth_waveform(
    wave: npt.NDArray, fill_value: float = 65535.0, moving_average_width: int = 3
) -> npt.NDArray:
    """Smooth a waveform using a moving average.

    Smoothes a waveform using a moving average. Output waveform retains NaN values from the
    original waveform.

    Args:
        wave (npt.NDArray): The input waveform.
        fill_value (float): Replaces all NaN values within the input waveform with this value.
        moving_average_width (int, optional): The number of consecutive values to use in the moving
            average. Defaults to 3.

    Raises:
        ValueError: Raised if the number of points used in the moving average is greater than the
            length of the input array.

    Returns:
        npt.NDArray: The output array with NaN values replaced.
    """

    if wave.size < moving_average_width:
        raise ValueError(
            "Length of input wave is less than the number of points in the moving average"
        )

    # Variables
    nan_indx: npt.NDArray = np.isnan(wave)  # index so we can retain NaN values in output wave
    # fill any NaNs with a filler value while smoothing to prevent NaNs being used in calculations
    filled_wave: npt.NDArray = np.nan_to_num(wave, nan=fill_value)

    smooth_wave: npt.NDArray = np.zeros(wave.size, dtype=float)

    moving_average_reach: int = moving_average_width // 2

    for i in range(filled_wave.size):
        if 0 + moving_average_reach <= i <= filled_wave.size - moving_average_reach:
            smooth_wave[i] = np.mean(
                filled_wave[i - moving_average_reach : i + moving_average_reach + 1]
            )
        else:
            smooth_wave[i] = filled_wave[i]

    smooth_wave[nan_indx] = np.nan  # replace the filled values with NaNs
    return smooth_wave
