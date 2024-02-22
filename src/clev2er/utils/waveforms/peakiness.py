"""Module for calculating the peakiness of a waveform.

peakiness(): Calculates the peakiness of a waveform for amplitude values above the nosie floor.


Author: Ben Palmer 
Date: 21 Feb 2024
"""

import numpy as np


def peakiness(waveform: np.ndarray, noise_floor_start: int, noise_floor_end: int) -> float:
    """calculate the peakiness of a waveform

    Calculates the peakiness of a waveform. The peakiness is defined as Pmax / Pmean, where
    Pmax is the maximum amplitude of the waveform, and Pmean is the mean amplitude above the noise
    floor. The noise floor is defined as the mean amplitude between two points on the waveform.
    Used for CS2 sea ice waveform discimination.

    Args:
        waveform: the waveform as a 1-d array.
        noise_floor_start: the bin number where the noise floor values starts
        noise_floor_end: the bin number where the noise floor values ends

    Returns:
        peakiness: the peakiness value of the waveform
    """

    peaky = np.nan

    noise_floor = np.nanmean(waveform[noise_floor_start:noise_floor_end])
    where_above_nf = np.where(waveform > noise_floor)[0]

    if where_above_nf.size > 0:
        above_nf = waveform[where_above_nf]

        nf_max = np.nanmax(above_nf)
        nf_mean = np.nanmean(above_nf)

        peaky = nf_max / nf_mean

    return peaky
