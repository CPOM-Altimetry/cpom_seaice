"""pytest for clev2er.utils.cs2.retrackers.gaussian_exponential_retracker.py

Author: Ben Palmer
Date: 01 Mar 2024
"""

# imports

import logging

import numpy as np
import numpy.typing as npt
import pytest
from numpy._typing._array_like import NDArray

from clev2er.utils.cs2.retrackers.gaussian_exponential_retracker import (
    gauss_plus_exp_tracker,
)

# logging set up

logger = logging.getLogger(__name__)

# fixtures


@pytest.fixture
def test_waveform() -> npt.NDArray:
    """Pytest fixture containing the test waveform

    Returns:
        npt.NDArray: test waveform
    """
    test_wave = np.asarray(
        [
            8.0,
            11.0,
            8.0,
            12.0,
            8.0,
            11.0,
            9.0,
            14.0,
            9.0,
            15.0,
            9.0,
            17.0,
            9.0,
            19.0,
            8.0,
            22.0,
            9.0,
            24.0,
            12.0,
            28.0,
            9.0,
            31.0,
            9.0,
            35.0,
            11.0,
            39.0,
            10.0,
            51.0,
            11.0,
            55.0,
            13.0,
            68.0,
            17.0,
            82.0,
            17.0,
            110.0,
            25.0,
            135.0,
            33.0,
            193.0,
            51.0,
            285.0,
            88.0,
            474.0,
            170.0,
            873.0,
            368.0,
            2805.0,
            1754.0,
            18283.0,
            65535.0,
            49310.0,
            10210.0,
            6463.0,
            2419.0,
            1984.0,
            1350.0,
            1211.0,
            775.0,
            856.0,
            558.0,
            546.0,
            530.0,
            450.0,
            376.0,
            324.0,
            340.0,
            260.0,
            222.0,
            270.0,
            252.0,
            159.0,
            143.0,
            198.0,
            158.0,
            143.0,
            161.0,
            112.0,
            133.0,
            150.0,
            173.0,
            146.0,
            153.0,
            125.0,
            126.0,
            111.0,
            118.0,
            88.0,
            108.0,
            120.0,
            101.0,
            73.0,
            89.0,
            81.0,
            90.0,
            78.0,
            99.0,
            99.0,
            126.0,
            114.0,
            111.0,
            81.0,
            88.0,
            71.0,
            73.0,
            64.0,
            53.0,
            50.0,
            44.0,
            35.0,
            42.0,
            40.0,
            41.0,
            36.0,
            36.0,
            29.0,
            38.0,
            24.0,
            29.0,
            27.0,
            31.0,
            26.0,
            31.0,
            48.0,
            53.0,
            45.0,
            37.0,
            24.0,
        ]
    )

    return test_wave


# pytest


def test_gauss_expo_retracker(
    test_waveform: NDArray,  # pylint: disable=redefined-outer-name
) -> None:
    """pytest for the gauss_plus_exp_tracker function

    Test plan:
        Get test wave with known retracking point
        Apply retracking function to find point
        Test that type of result is correct
        Test that result is correct

    Args:
        test_waveform (NDArray): pytest fixture for the input array

    Returns:
        None
    """

    logger.info("Testing gauss_plus_exp_tracker")

    retracking_point = gauss_plus_exp_tracker(test_waveform)

    assert isinstance(retracking_point, float), "Retracker does not return a float value"

    # atol is larger here than other closeness checks, different devices return slightly different
    # retracking values, but it should return a value be within range for this waveform
    assert np.isclose(
        retracking_point, 50.35911696865722, atol=0.5
    ), "Retracker did not return a value close to what is expected"
