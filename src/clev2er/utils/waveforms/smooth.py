""" Module for smoothing waveforms.

    smooth_waveform(): Smoothes the input waveform using an N-point moving average.

    Author: Ben Palmer
    Date: 22 May 2024
"""

from numpy import mean as np_mean  # prevents shadowing mean function
from numpy import ndarray


def smooth_waveform(waveform: ndarray, moving_avg_size: int = 3) -> ndarray:
    """Smoothes an input waveform using a moving average of size moving_avg_size.

    Args:
        waveform (ndarray): The input waveform array
        moving_avg_size (int, optional): The size of the moving average. Defaults to 3.

    Returns:
        ndarray: The smoothed waveform
    """
    reach = moving_avg_size // 2
    smoothed_wave = waveform.copy()
    for i in range(reach, waveform.size - reach):
        smoothed_wave[i] = np_mean(waveform[i - reach : i + reach + 1])
    return smoothed_wave
