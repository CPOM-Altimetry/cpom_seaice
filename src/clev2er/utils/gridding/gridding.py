"""Module for gridding point data into 2d grids.
grid_points_sum: function to add points to a 2d grid by adding points together
GriddedDataFile: Helper class for adding points to a netcdf containing gridded data
"""

import fcntl
import os
import signal
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.utils.gridding.locators import get_cell_indexes_from_lat_lon


def grid_points_sum(
    ilats: np.ndarray,
    ilons: np.ndarray,
    z: np.ndarray | float | int,
    grid_array: np.ndarray,
    where: Optional[np.ndarray] = None,
):
    """Function to add values to a 2d gridded array from lat lon values.

    Args:
        lats (np.ndarray): Latitude values of points
        lons (np.ndarray): Longitude values of points
        z (np.ndarray | float | int): Values to be added (accepts singular values)
        grid_array (np.ndarray): Grid to be added to
        where (np.ndarray, optional): Option to index array using boolean or index array.
            Defaults to None.
    """
    if where is None:
        where = np.full_like(ilats, True).astype(bool)
    if not isinstance(z, np.ndarray):
        z = np.full_like(ilats, z)
    for t_ilat, t_ilon, t_z, t_w in zip(ilats, ilons, z, where):
        if t_w:
            if np.isnan(t_z):
                continue
            if np.isnan(grid_array[t_ilat, t_ilon]):
                grid_array[t_ilat, t_ilon] = 0
            grid_array[t_ilat, t_ilon] += t_z


def grid_points_equals(
    ilats: np.ndarray,
    ilons: np.ndarray,
    z: np.ndarray | float | int,
    grid_array: np.ndarray,
    where: Optional[np.ndarray] = None,
):
    """Function to set values to a 2d gridded array from lat lon values.

    Args:
        lats (np.ndarray): Latitude values of points
        lons (np.ndarray): Longitude values of points
        z (np.ndarray | float | int): Values to be added (accepts singular values)
        grid_array (np.ndarray): Grid to be added
        where (np.ndarray, optional): Option to index array using boolean or index array.
            Defaults to None.
    """
    if where is None:
        grid_array[ilats, ilons] = z
    else:
        if not isinstance(z, np.ndarray):
            z = np.full_like(ilats, z)
        grid_array[ilats[where], ilons[where]] = z[where]


@dataclass
class VariableSpec:
    """Class to hold specification information for variables input to the
    GriddedDataFile class
    """

    name: str
    dtype: str
    dimensions: tuple[str, ...]
    init_value: Any
    compression: Optional[str] = "zlib"


class GriddedDataFile(AbstractContextManager):
    """Class for created a gridded data file for gridding and passing data between CLEV2ER stages

    This is also a context manager, so can be used in 'with' statements and will save data
    upon leaving context.

    Example:
    with GriddedDataFile(...) as this_grid_file:
        this_grid_file.add_attributes({...})
        this_grid_file.grid_points(
                                coordinates={"lat":..., "lon":...},
                                data={"var1":...},
                                conditions={"var1":...}
                            )

    # Closes here and saves data to file
    """

    # pylint:disable=too-many-instance-attributes
    # pylint:disable=unspecified-encoding
    # pylint:disable=consider-using-with

    def __init__(
        self,
        variables: list[VariableSpec],
        filename: str,
        nrows: int,
        ncols: int,
        timeout: int = 30,
    ):
        # pylint:disable=too-many-arguments
        self.variables = variables
        self.filename = filename
        self.nrows = nrows
        self.ncols = ncols
        self.timeout = timeout
        self.lock_file_path = filename + ".lock"
        self.lock_file = None

        self.attributes: dict = {}

        self._acquire_lock()

        try:
            if os.path.exists(self.filename):
                self.nc, self.arrays = self._load_existing(self.filename)
            else:
                self.nc, self.arrays = self._create_netcdf(self.filename)
        except:
            self._release_lock()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)

    def _acquire_lock(self):
        """Acquire lock file so that other processes don't attempt to write at the same time"""

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Lock acquisition timed out after {self.timeout}s")

        self.lock_file = open(self.lock_file_path, "w")

        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)  # Blocking
            signal.alarm(0)  # Cancel alarm
        except ValueError:
            signal.alarm(0)  # Ensure alarm is cancelled
            self.lock_file.close()
            raise

    def _release_lock(self):
        """Release the file lock"""
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except ValueError:
                pass

    def _create_netcdf(self, filepath: str):
        output_nc: Dataset = Dataset(filepath, mode="w")  # type: ignore
        output_nc.createDimension("lat", self.nrows)
        output_nc.createDimension("lon", self.ncols)

        for var in self.variables:
            output_nc.createVariable(
                var.name, var.dtype, var.dimensions, compression=var.compression
            )

        arrays = {
            var.name: np.full(
                (self.nrows, self.ncols), dtype=np.dtype(var.dtype), fill_value=var.init_value
            )
            for var in self.variables
        }
        return output_nc, arrays

    def _load_existing(self, filepath: str):
        output_nc: Dataset = Dataset(filepath, mode="a")
        arrays = {var.name: output_nc[var.name][:].data for var in self.variables}
        return output_nc, arrays

    def _close_file(self):
        # save arrays to netcdf
        for var_name, array in self.arrays.items():
            self.nc[var_name][:] = array

        for attr_name, attr_val in self.attributes.items():
            setattr(self.nc, attr_name, attr_val)

        self.nc.close()

    def get_attributes(self) -> dict:
        """Gets attributes as a dictionary

        Returns:
            dict: attributes as key value pairs
        """
        return {key: getattr(self.nc, key) for key in self.nc.ncattrs()}

    def add_attributes(self, attributes: Dict[str, Any]):
        """Adds attributes to the output NC file

        Args:
            attributes (Dict[str, Any]): Dictionary of attributes as key-value pairs
        """
        for key, value in attributes.items():
            setattr(self.nc, key, value)

    def grid_points(
        self,
        coordinates: Dict[str, np.ndarray],
        data: Dict[str, np.ndarray | int | float],
        conditions: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Grids data and adds it to the grid arrays within the NC file.

        Args:
            coordinates (Dict[str, np.ndarray]): Coordinates of each point. Must contain
                two arrays - "lat" and "lon".
            data (Dict[str, np.ndarray  |  int  |  float]): Variables for each point.
                Each must have the same dimensions as input coordinates.
            conditions (Dict[str, np.ndarray]): Conditions that variables have to meet
                in order to be gridded. Dictionary of {variable_name:boolean array}

        Raises:
            RuntimeError: Raised if either "lat" or "lon" isn't within the
                coordinate parameters
            RuntimeError: Raised if there are keys in the input data that
                don't have a specification
            RuntimeError: Raised if there are keys in the conditions argument
                that don't exist in data
        """
        if "lat" not in coordinates or "lon" not in coordinates:
            raise RuntimeError("Need both lat and lon values in coordinates")

        if not set(data.keys()).issubset(self.arrays.keys()):
            raise RuntimeError(
                f"""Provided data that isn't within variable specifications! 
                keys={set(data.keys()).difference(self.arrays.keys())}"""
            )

        if conditions and not set(conditions.keys()).issubset(data.keys()):
            raise RuntimeError(
                f"""Provided keys for conditions that are not included in data!
                keys={set(conditions.keys())}"""
            )

        ilats, ilons = get_cell_indexes_from_lat_lon(coordinates["lat"], coordinates["lon"])

        for var_name, var_data in data.items():
            if not isinstance(var_data, np.ndarray):
                var_data = np.full(
                    (len(coordinates["lat"])),
                    fill_value=var_data,
                    dtype=self.arrays[var_name].dtype,
                )
            mask = np.ones_like(var_data, dtype=np.bool_)

            # Don't grid any nans
            var_is_nan = np.isnan(var_data)
            if np.any(var_is_nan):
                mask &= ~var_is_nan

            # If there are any conditions given, mask for them
            if conditions and var_name in conditions:
                mask &= conditions[var_name]

            grid_points_sum(ilats, ilons, var_data, self.arrays[var_name], where=mask)

    def close(self):
        """Closes the file and releases lock"""
        try:
            self._close_file()
        finally:
            self._release_lock()
