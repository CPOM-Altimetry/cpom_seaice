"""Module for the gaussian plus exponential retracker (or Giles' retracker)

gauss_plus_exp_tracker(): the retracking function which fits a gaussian plus exponential function
    to the input waveform to find the retracking point.
_GaussPlusExp(): the gaussian exponential function used by gauss_plus_exp_tracker

Author: Ben Palmer
Date: 01 Mar 2024
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


def gauss_plus_exp_tracker(waveform: npt.NDArray, max_iterations: int = 3000) -> float:
    """Gaussian plus exponential retracker, or Giles' retracker.

    Retracks waveforms using a gaussian plus exponential function. Uses Levenberg-Marquardt
    non-linear least-squares method to fit the function to the waveform. Returns the location of the
    maximum amplitude of the fitted wave as the retracking point.

    Args:
        waveform (npt.NDArray): Input waveform

    Raises:
        RuntimeError: Raised if GaussPlusExp function encounters an error while fitting.

    Returns:
        float: The retracking point
    """

    warnings.simplefilter("ignore", RuntimeWarning, 84)

    x: npt.NDArray = np.arange(waveform.size).astype(float)

    tracking_point = np.nan

    x0: np.intp = np.argmax(waveform)
    a: float = waveform[x0]

    try:
        popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
            _gauss_plus_exp, x, waveform[:], p0=[a, x0, 1, 1], maxfev=max_iterations, method="lm"
        )
        tracking_point = popt[1]  # Update tracking point if successful, otherwise returns NaN
    except RuntimeError:
        pass

    return tracking_point


def _gauss_plus_exp(x: npt.NDArray, a: float, x0: np.intp, k: float, sigma: float) -> npt.NDArray:
    """Gaussian Plus Exponential distribution.

    Args:
        x (npt.NDArray): _description_
        a (float): maximum amplitude of the distribution
        x0 (np.intp): x when amplitude = maximum
        k (float): rate of decay for the decaying exponential function
        sigma (float): standard deviation of the Gaussian function

    Returns:
        npt.NDArray: The values of the funciton
    """

    xb = k * (sigma**2)
    a2 = ((5 * k * sigma) - (4 * np.sqrt(k * xb))) / (2 * sigma * xb * np.sqrt(k * xb))
    a3 = ((2 * np.sqrt(k * xb)) - (3 * k * sigma)) / (2 * sigma * (xb**2) * np.sqrt(k * xb))

    y = np.piecewise(
        x,
        [x <= x0, (x > x0) & (x < (x0 + xb)), x > (x0 + xb)],
        [
            # linking function
            lambda x: a * np.exp(-(((x - x0) / sigma) ** 2)),
            # function to simulate the leading edge
            lambda x: -((a3 * ((x - x0) ** 3) + a2 * ((x - x0) ** 2) + ((x - x0) / sigma)) ** 2),
            # function to simulate the trailing edge
            # NOTE: This is the one used in Giles et al from 2007. Tilling et al in 2017 has minor
            # differences to this one, but the original is used here.
            lambda x: a * np.exp(-(np.sqrt(k * (x - x0)) ** 2)),
        ],
    )

    return y
