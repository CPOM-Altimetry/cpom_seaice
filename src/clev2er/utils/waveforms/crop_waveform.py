"""Module for cropping waveforms from L1B files.

crop_waveform(): Crops a given waveform to a set length with a set number of bins before the bin 
                with maximum amplitude.

Author: Ben Palmer 
Date: 22 Feb 2024
"""

import numpy as np


def crop_waveform(waveform: np.ndarray, length_before_max: int, cropped_length: int) -> np.ndarray:
    """Crops a waveform around the bin with maximum amplitude.

    Crops a waveform to a given length with a given number of bins before the maximum amplitude.

    Args:
        waveform: a numpy 1-d array that contains the waveform.
        length_before_max: the number of bins before the bin with the max amplitude value on the
                        waveform.
        cropped_length: the total length of the cropped waveform including length_before_max.

    Returns:
        A numpy array of the cropped waveform.
    """

    b_max = np.nanargmax(waveform)
    cropped_waveform = waveform[
        b_max - length_before_max : b_max + (cropped_length - length_before_max)
    ]
    return cropped_waveform
