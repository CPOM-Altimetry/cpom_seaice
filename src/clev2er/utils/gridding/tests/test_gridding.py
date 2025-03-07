"""Pytest file for contents of clev2er.utils.gridding.gridding"""

import os
import tempfile

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.utils.gridding.gridding import (
    GriddedDataFile,
    VariableSpec,
    grid_points_equals,
    grid_points_sum,
)


def test_grid_points_sum():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Add variables to an empty grid
    - Test if grid is as expected
    """

    array = np.zeros((3, 3))
    ilats = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ilons = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])

    grid_points_sum(ilats, ilons, 1, array)

    assert np.array_equal(
        np.ones_like(array), array
    ), f"grid_points_sum did not return expected results\n{array}"


def test_grid_points_sum_cond():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Add variables to an empty grid with condition
    - Test if grid is as expected
    """
    array = np.zeros((3, 3))
    ilats = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ilons = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    vals = np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4])
    cond = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.bool_)

    grid_points_sum(ilats, ilons, vals, array, cond)

    assert np.array_equal(
        [[0, 4, 0], [4, 0, 4], [0, 4, 0]], array
    ), f"grid_points_sum did not return expected results when used with condition\n{array}"


def test_grid_points_sum_nan():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Add variables to a NaN grid
    - Test if grid is as expected
    """
    array = np.full((3, 3), fill_value=np.nan)
    ilats = np.asarray([1])
    ilons = np.asarray([1])
    vals = 2

    grid_points_sum(ilats, ilons, vals, array)

    assert np.array_equal(
        array,
        np.asarray([[np.nan, np.nan, np.nan], [np.nan, 2, np.nan], [np.nan, np.nan, np.nan]]),
        equal_nan=True,
    ), f"grid_points_sum did not return expected results when adding to NaN grid\n{array}"


def test_grid_points_sum_nan_cond():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Add variables to a NaN grid with condition
    - Test if grid is as expected
    """
    array = np.full((3, 3), fill_value=np.nan)
    ilats = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ilons = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    vals = np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4])
    cond = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.bool_)

    grid_points_sum(ilats, ilons, vals, array, cond)

    assert np.array_equal(
        [[np.nan, 4, np.nan], [4, np.nan, 4], [np.nan, 4, np.nan]], array, equal_nan=True
    ), f"""grid_points_sum did not return expected results when used with condition on NaN grid
    {array}"""


def test_grid_points_equals():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Set variables to an empty grid
    - Test if grid is as expected
    """
    array = np.zeros((3, 3)) * np.nan
    ilats = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ilons = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    vals = 1
    cond = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.bool_)

    print(ilats, ilons, vals, cond)

    grid_points_equals(ilats, ilons, vals, array, cond)

    assert np.array_equal(
        [[np.nan, 1, np.nan], [1, np.nan, 1], [np.nan, 1, np.nan]], array, equal_nan=True
    ), f"grid_points_equals did not return expected results\n{array}"


def test_gridded_data_file():
    """Pytest for GriddedDataFile

    Plan:
    - Define test variable
    - Create temporary directory
    - Create GriddedDataFile instance
    - Add an attribute
    - Add data
    - Close instance
    - Open file using netcdf4 library
    - Check attribute exists
    - Check data exists
    - Check data is correct
    """
    variables = [
        VariableSpec(name="test", dtype="i4", dimensions=("lat", "lon"), init_value=0),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        test_filename = os.path.join(temp_dir, "temp_file.nc")
        with GriddedDataFile(variables=variables, filename=test_filename, nrows=1, ncols=1) as gdf:
            gdf.add_attributes({"test_attr": "hello"})
            gdf.grid_points(
                {"lat": np.asarray([40]), "lon": np.asarray([-180])}, {"test": np.asarray([5])}
            )

        nc = Dataset(test_filename, "r")
        assert "test_attr" in nc.ncattrs(), "Attributes not added successfully"

        assert nc.test_attr == "hello", "Attribute value not as expected"

        assert "test" in nc.variables.keys(), "Variables not successfully added"

        test_var = nc["test"]

        assert np.shape(test_var) == (
            1,
            1,
        ), f"Test variable is not the correct shape. Shape={np.shape(test_var)}"

        assert (
            test_var[0, 0] == 5
        ), f"Value of test variable is not as expected. Value={test_var[0,0]}"
