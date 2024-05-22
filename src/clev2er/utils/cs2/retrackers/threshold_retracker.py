"""Module for the threshold retracker used for CS2 sea ice waveforms.

threshold_retrack(): Threshold retracker that finds a retracking point at X% of the amplitude after
                the first peak

Author: Ben Palmer
Date: 23 Feb 2024
"""

import numpy as np
from numpy import typing as npt


def threshold_retracker(waveform: npt.NDArray, threshold: float) -> float:
    """Function for threshold retracker.

    Finds a retracking point at X% of the amplitude of first peak of the waveform.
    The retracking point will be between:
    - 20% of the max amplitude
    - the max amplitude of the first peak

    Args:
        waveform (npt.NDArray): The waveform as a 1-d array
        threshold (float): The threshold as a float (e.g. 0.7 for 70%)

    Raises:
        ValueError: Raised if the threshold value provided is not between 0 and 1

    Returns:
        float: The retracking bin
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold should be between 0 and 1, received {threshold}")

    # NOTE: Contains a Python-y version and Andy's version.

    # NOTE: Pythony version below.

    # bin_max = np.nanargmax(waveform)
    # amp_max = waveform[bin_max]
    # if bin_max <= 0:
    #     return np.nan

    # threshold_20 = amp_max * 0.2
    # idx_gt_threshold_20 = np.where(waveform >= threshold_20)[0]
    # idx_before_peak = np.where(idx_gt_threshold_20 <= bin_max)[0]
    # idx_gt_20_before_peak = idx_gt_threshold_20[idx_before_peak]

    # indx = 0  # initialise index variable (or else pylint raises issue)

    # for indx in idx_gt_20_before_peak:
    #     if indx < waveform.size - 1 and waveform[indx - 1] < waveform[indx] > waveform[indx + 1]:
    #         break

    # first_peak_amp = waveform[indx]

    # threshold_amp = threshold * first_peak_amp

    # idx_gt_threshold = np.where(waveform > threshold_amp)[0]

    # if idx_gt_threshold.size <= 0:
    #     return np.nan

    # x2 = np.min(idx_gt_threshold)  # find first bin where amp > threshold

    # NOTE: Andy's version below.

    idx = 0
    amp_max = np.max(waveform)
    valid_amp = amp_max * 0.2

    while waveform[idx] < valid_amp and idx < waveform.size:
        idx += 1

    while waveform[idx] < waveform[idx + 1] and idx < waveform.size:
        idx += 1

    threshold_amp = waveform[idx] * threshold

    x2 = 0
    while waveform[x2] < threshold_amp:
        x2 += 1
    x1 = x2 - 1

    if x2 == 0 or waveform[x2] <= waveform[x1]:
        return 0.0

    return x1 + ((threshold_amp - waveform[x1]) / (waveform[x2] - waveform[x1]))
