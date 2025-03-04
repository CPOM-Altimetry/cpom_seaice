"""Pytest file for contents of clev2er.utils.gridding.gridding"""

import os
import tempfile

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.utils.gridding.gridding import (
    GriddedDataFile,
    VariableSpec,
    grid_points_sum,
)


def test_grid_points_sum():
    """Pytest for grid_points_sum

    Plan:
    - Initialise test variables
    - Add variables to an empty grid
    - Test if grid is as expected
    - Add variables to another empty grid with condition
    - Test if grid is as expected
    """
    array = np.zeros((3, 3))
    ilats = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ilons = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    vals = np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4])
    cond = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.bool_)

    print(ilats, ilons, vals, cond)

    grid_points_sum(ilats, ilons, vals, array, cond)

    assert np.array_equal(
        [[0, 4, 0], [4, 0, 4], [0, 4, 0]], array
    ), f"grid_points_sum did not return expected results\n{array}"

    array_2 = np.zeros((3, 3))
    grid_points_sum(ilats, ilons, 1, array_2)

    assert np.array_equal(
        np.ones_like(array_2), array_2
    ), f"grid_points_sum did not return expected results\n{array_2}"


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
    variables = [VariableSpec("test", "i4", ("lat", "lon"))]

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
