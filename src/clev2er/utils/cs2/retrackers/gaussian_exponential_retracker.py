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
from scipy.optimize import OptimizeWarning, curve_fit


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

    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", OptimizeWarning)

    x: npt.NDArray = np.arange(waveform.size).astype(float)

    tracking_point = np.nan

    x0: np.intp = np.argmax(waveform)
    a: float = waveform[x0]
    sigma: float = 1
    k: float = 1

    for i in range(waveform.size):
        if waveform[i] >= a / 2.0:
            if x0 == i:
                sigma = 1 / 2.35
                k = 0.5
            else:
                sigma = float(2.0 * (x0 - i) / 2.35)
                k = float((x0 - i) / sigma**2)

            break

    try:
        popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
            _gauss_plus_exp,
            x,
            waveform,
            p0=[a, x0, sigma, k],
            maxfev=max_iterations,
            method="lm",
            jac=_jacgexp,
        )
        tracking_point = popt[1]  # Update tracking point if successful, otherwise returns NaN
    except RuntimeError:
        pass

    return tracking_point


def _gauss_plus_exp(x: npt.NDArray, a: float, x0: np.intp, sigma: float, k: float) -> npt.NDArray:
    """Gaussian Plus Exponential distribution.

    Args:
        x (npt.NDArray): Array of x values
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

    x[x == 128] = 44.5
    x[x == 129] = 45.5

    def f_gaussian(x):
        return (x - x0) / sigma

    def f_linking(x):
        return (a3 * ((x - x0) ** 3)) + (a2 * ((x - x0) ** 2)) + ((x - x0) * 1 / sigma)

    def f_exponential(x):
        return np.sqrt(k * (x - x0))

    def f_p(f):
        return a * np.exp(-(f**2))

    y = np.piecewise(
        x,
        [x <= x0, (x > x0) & (x < (x0 + xb)), x > (x0 + xb)],
        [
            # function to simulate the leading edge
            lambda x: f_p(f_gaussian(x)),
            # linking function
            lambda x: f_p(f_linking(x)),
            # function to simulate the trailing edge
            # NOTE: This is the one used in Giles et al from 2007. Tilling et al in 2017 has minor
            # differences to this one, but the original is used here.
            lambda x: f_p(f_exponential(x)),
        ],
    )

    return y


def _jacgexp(x, a, x0, sigma, k):
    # pylint: disable=too-many-locals
    m = 4
    n = len(x)

    jac = np.zeros((n, m))

    tb = k * sigma * sigma

    c = np.sqrt(tb * k)
    bot = 2.0 * sigma * tb * c
    a2 = -((-5.0 * k * sigma) + (4.0 * c)) / bot
    bot *= tb
    a3 = -((3.0 * k * sigma) - (2.0 * c)) / bot

    da2dp2 = -3.0 / (2.0 * sigma * sigma * tb)
    da2dp3 = -1.0 / (2.0 * sigma * k * tb)
    da3dp2 = 5.0 / (2.0 * sigma * sigma * tb * tb)
    da3dp3 = 1.0 / (1.0 * sigma * k * tb * tb)

    for i in range(n):
        di = x[i]
        if di == 128:
            di = 44.5
        elif di == 129:
            di = 45.5

        if di <= x0:
            tmp1 = (di - x0) / sigma
            tmp2 = np.exp(-tmp1 * tmp1)
            j0 = tmp2
            j1 = 2.0 * tmp1 * a * tmp2 / sigma
            j2 = j1 * tmp1
            j3 = 0.0
        elif x0 < di < x0 + tb:
            tmp0 = di - x0
            tmp1 = (a3 * tmp0**3) + (a2 * tmp0**2) + (tmp0 / sigma)
            tmp2 = np.exp(-tmp1 * tmp1)
            tmp3 = (-3.0 * a3 * tmp0**2) + (-2.0 * a2 * tmp0) - (1.0 / sigma)
            tmp4 = (da3dp2 * tmp0**3) + (da2dp2 * tmp0**2) - (tmp0 / (sigma * sigma))
            tmp5 = (da3dp3 * tmp0**3) + (da2dp3 * tmp0**2)
            j0 = tmp2
            j1 = -2.0 * tmp1 * a * tmp2 * tmp3
            j2 = -2.0 * tmp1 * a * tmp2 * tmp4
            j3 = -2.0 * tmp1 * a * tmp2 * tmp5
        else:
            tmp1 = k * (x0 - di)
            tmp2 = np.exp(tmp1)
            j0 = tmp2
            j1 = k * a * tmp2
            j2 = 0.0
            j3 = (x0 - di) * a * tmp2

        jac[i, 0] = j0
        jac[i, 1] = j1
        jac[i, 2] = j2
        jac[i, 3] = j3

    return jac


def _get_fit_qual(a, x0, waveform, best_fit_waveform, n_bins=5):
    b_min = x0 - n_bins

    tmp1 = 0
    tmp2 = 0

    for i in range(b_min, x0):
        tmp1 += (waveform[i] - best_fit_waveform[i]) ** 2
        tmp2 += a**2

    return 1000 * np.sqrt(tmp1 / tmp2) if tmp2 > 0 else -999.999
