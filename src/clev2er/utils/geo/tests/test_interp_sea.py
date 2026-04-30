"""Module for testing the clev2er.utils.geo.interp_sea.py module

Author: Ben Palmer
Date: 18 Mar 2024
"""

import numpy as np

from clev2er.utils.geo.interp_sea import earth_dist, interp_sea_regression, regress


def test_interp_sea_regression() -> None:
    """Test for interp_sea_regression

    Test plan:
    Generate fake lat, lon, sea_level and lead arrays.
    Check that output is an array of floats,
    Check that output is the same size as inputs.
    Check that output contains values within the accepted ranges.
    """
    lats = np.arange(60, 80, 0.1)
    lons = np.arange(200, 220, 0.1)
    sea_level = np.arange(1000, 1200)

    interp_sla = interp_sea_regression(lats, lons, sea_level, window_size=100000)

    assert isinstance(
        interp_sla, np.ndarray
    ), f"Output is not a numpy ndarray - found {type(interp_sla)}"
    assert (
        "float" in str(interp_sla.dtype).lower()
    ), f"Output array does not contain float values - found {interp_sla.dtype}"
    assert lats.shape[0] == interp_sla.shape[0], "Output is not the same length as input"
    assert np.nanmax(interp_sla) < np.nanmax(sea_level) and np.nanmin(interp_sla) > np.nanmin(
        sea_level
    ), "Output array contains values outside of given range"


def test_regress() -> None:
    """Test for regress

    Test plan:
    Make fake points that have known gradient and y-intercept
    Use regress function
    Check results are floats
    Check results are what is expected
    """

    x = np.asarray([-1, 0, 1, 2])
    y = np.asarray([0, 1, 2, 3])

    res = regress(x, y)
    m, c = res

    assert isinstance(m, np.float_) and isinstance(
        c, np.float_
    ), f"Output values are not floats, got type {(type(m), type(c))}"

    assert res == (1, 1), f"Output values are not as expected. Expecting (1,1), got {res}."


def test_earth_dist() -> None:
    """Test for earth_dist

    Test plan:
    Create two points which are a set distance apart
    Use earth_dist to find distance
    Check result is a float
    Check result is as expected/close

    """

    lat1 = lon1 = 45
    lat2 = lon2 = 46
    assumed_dist = 135786

    predicted_dist = earth_dist(lat1, lon1, lat2, lon2)

    assert isinstance(
        predicted_dist, np.float_
    ), f"Output value is not a float, got type {type(predicted_dist)}"

    assert (
        np.abs(predicted_dist - assumed_dist) < 1000
    ), f"Output values are not as expected. Expecting {assumed_dist}+-1000m, got {predicted_dist}."
