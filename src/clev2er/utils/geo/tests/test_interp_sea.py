""" Module for testing the clev2er.utils.geo.interp_sea.py module

    Author: Ben Palmer
    Date: 18 Mar 2024
"""

import numpy as np

from clev2er.utils.geo.interp_sea import interp_sea_regression


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
    leads = np.random.randint(0, 2, lats.size).astype(bool)

    interp_sla = interp_sea_regression(lats, lons, sea_level, lead_index=leads, window_size=100000)

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
