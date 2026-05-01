"""clev2er.algorithms.seaice.alg_grid_average.py


Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Grids points using latitude and longitude by finding all points within a set radius of each
grid cell. Uses the mean of found points.
Uses a grid defined by an external file.

#Main initialization (init() function) steps/resources required

Get config parameters
set up projections and coordinate transform
Load grid from file
set up kdtree query points from grid coordinates
load variables to process

#Main process() function steps

calculate along-track x/y from lon/lat
create tree from points
init variable arrays and input data from file
find indices of nearest points to each grid cell
for each list of indices:
    calculate statistics
    save to variable arrays
save all data to shared_dict

#Main finalize() function steps

None

#Contribution to shared_dict

shared_dict["grid_lat"]: Central latitude for each grid cell
shared_dict["grid_lon"]: Central longitude for each grid cell
shared_dict["grid_x"]: Central x for each grid cell
shared_dict["grid_y"]: Central y for each grid cell
shared_dict["grid_mask"]: Flag for each grid cell where True is an invalid gridcell

#Requires from shared_dict

None

Author: Ben Palmer
Date: 20 Feb 2026
"""

import os
from typing import Tuple

import cartopy.crs as ccrs
import numpy as np
import pyproj as proj
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from sklearn.neighbors import BallTree

from clev2er.algorithms.base.base_alg import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    # pylint:disable=too-many-instance-attributes
    """CLEV2ER Algorithm class

    contains:
            .log (Logger) : log instance that must be used for all logging, set by BaseAlgorithm
            .config (dict) : configuration dictionary, set by BaseAlgorithm

        functions that need completing:
            .init() : Algorithm initialization function (run once at start of chain)
            .process(l1b,shared_dict) : Algorithm processing function (run on every L1b file)
            .finalize() :   Algorithm finalization/closure function (run after all chain
                            processing completed)

        Inherits from BaseAlgorithm which handles interaction with the chain controller run_chain.py

    """

    def init(self) -> Tuple[bool, str]:
        """Algorithm initialization function

        Add steps in this function that are run once at the beginning of the chain
        (for example loading a DEM or Mask)

        Returns:
            (bool,str) : success or failure, error string

        Test for KeyError or OSError exceptions and raise them if found
        rather than just returning (False,"error description")

        Raises:
            KeyError : for keys not found in self.config
            OSError : for any file related errors

        Note:
        - retrieve required config data from self.config dict
        - log using self.log.info(), or self.log.error() or self.log.debug()

        """
        # pylint:disable=pointless-string-statement
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ 
        Get config parameters
        Check that output directory exists
        """

        self.input_crs = ccrs.CRS(self.config["alg_grid_average"]["input_projection"])
        self.target_crs = ccrs.CRS(self.config["alg_grid_average"]["output_projection"])
        self.xy_transform = proj.Transformer.from_crs(
            self.input_crs, self.target_crs, always_xy=True
        )
        self.search_radius = self.config["alg_grid_average"]["search_radius"]

        grid_definition_file: str = self.config["alg_grid_average"]["grid_file"]

        if not os.path.exists(grid_definition_file):
            raise FileNotFoundError(  # pylint:disable=raising-format-tuple
                "Cannot find grid file %s", grid_definition_file
            )

        if grid_definition_file.endswith(".npz"):
            grid_data = np.load(grid_definition_file)
            self.grid_lats = grid_data["lats"]
            self.grid_lons = grid_data["lons"]
            grid_mask = grid_data["mask"]
            self.output_shape = self.grid_lats.shape
        else:
            raise RuntimeError("Grid definition file is in an unknown format")

        (  # pylint:disable=unpacking-non-sequence
            self.grid_x,
            self.grid_y,
        ) = self.xy_transform.transform(self.grid_lons.flatten(), self.grid_lats.flatten())

        self.query_points = np.transpose([self.grid_x.flatten(), self.grid_y.flatten()])

        self.invalid_points = ((self.grid_lats <= 60.0) | (self.grid_lats >= 88.0)) | (grid_mask)

        self.variables = self.config["alg_grid_average"]["variables"]

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
        # pylint: disable=too-many-statements
        """Main algorithm processing function, called for every L1b file

        Args:
            l1b (Dataset): input l1b file dataset (constant)
            shared_dict (dict): shared_dict data passed between algorithms. Use this dict
                                to pass algorithm results down the chain or read variables
                                set by other algorithms.

        Returns:
            Tuple : (success (bool), failure_reason (str))
            ie
            (False,'error string'), or (True,'')

        Note:
        - retrieve required config data from self.config dict (read-only)
        - retrieve data from other algorithms from shared_dict
        - add results,variables from this algorithm to shared_dict
        - log using self.log.info(), or self.log.error() or self.log.debug()

        """

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # /    down the chain in the 'shared_dict' dict     /
        # -------------------------------------------------------------------

        """ 
        
        """

        # transform lat/lon to x/y
        point_x, point_y = self.xy_transform.transform(
            l1b["sat_lon"][:].data.flatten(), l1b["sat_lat"][:].data.flatten()
        )

        tree_points = np.transpose([point_x, point_y])

        tree = BallTree(tree_points, metric="euclidean")

        indices, distances = tree.query_radius(
            self.query_points, r=self.search_radius, return_distance=True
        )

        gridded_data = {}
        tree_values = {}

        for var_name in self.variables:
            if var_name not in shared_dict.keys():
                raise RuntimeError(f"Variable {var_name} not found in shared_dict.")

            variable_data = {
                "values": np.zeros(len(indices)),
                "std": np.zeros(len(indices)),
                "n_points": np.zeros(len(indices)),
                "mean_distance": np.zeros(len(indices)),
                "distance_from_cog": np.zeros(len(indices)),
            }

            gridded_data[var_name] = variable_data

            tree_values[var_name] = shared_dict[var_name].flatten()

        for i, (ind, dist) in enumerate(zip(indices, distances)):
            print(f"{i}/{len(indices)} ({i*100//len(indices)}%)", end="\r")

            if len(ind) <= 1:
                continue

            if not (dist < self.search_radius).all():
                raise RuntimeError("Not all points within radius")

            for var_name in self.variables:
                vals = tree_values[var_name][ind]
                invalid_values = np.isnan(vals)
                if invalid_values.all() or len(ind) <= 1:
                    gridded_data[var_name]["values"][i] = gridded_data[var_name]["std"][
                        i
                    ] = gridded_data[var_name]["n_points"][i] = gridded_data[var_name][
                        "mean_distance"
                    ][
                        i
                    ] = gridded_data[
                        var_name
                    ][
                        "distance_from_cog"
                    ][
                        i
                    ] = 0
                else:
                    valid_vals = vals[~invalid_values]
                    gridded_data[var_name]["values"][i] = np.mean(valid_vals)
                    gridded_data[var_name]["std"][i] = np.std(valid_vals)
                    gridded_data[var_name]["n_points"][i] = len(valid_vals)
                    gridded_data[var_name]["mean_distance"][i] = np.mean(dist)
                    mean_x = np.mean(point_x[ind][~invalid_values])
                    mean_y = np.mean(point_y[ind][~invalid_values])
                    gridded_data[var_name]["distance_from_cog"][i] = np.sqrt(
                        (self.grid_x[i] - mean_x) ** 2 + (self.grid_y[i] - mean_y) ** 2
                    )

        invalid_mask = self.invalid_points.flatten()

        for var_name in self.variables:
            # mask out invalid gridcells
            gridded_data[var_name]["values"][invalid_mask] = 0
            gridded_data[var_name]["n_points"][invalid_mask] = 0
            gridded_data[var_name]["std"][invalid_mask] = 0
            gridded_data[var_name]["distance_from_cog"][invalid_mask] = 0

            # add to shared_dict with the variable name attached
            shared_dict[var_name + "_grid"] = gridded_data[var_name]["values"]
            shared_dict[var_name + "_std"] = gridded_data[var_name]["std"]
            shared_dict[var_name + "_n_points"] = gridded_data[var_name]["n_points"]
            shared_dict[var_name + "_distance_from_cog"] = gridded_data[var_name][
                "distance_from_cog"
            ]

        shared_dict["grid_lat"] = self.grid_lats
        shared_dict["grid_lon"] = self.grid_lons
        shared_dict["grid_x"] = self.grid_x
        shared_dict["grid_y"] = self.grid_y
        shared_dict["grid_mask"] = self.invalid_points

        # -------------------------------------------------------------------
        # Returns (True,'') if success
        return (success, error_str)

    def finalize(self, stage: int = 0) -> None:
        """Algorithm finalization function - called after all processing completed

        Can be used to clean up/free resources initialized in the init() function

        Args:
            stage (int, optional):  this sets the stage when this function is called
                                    by the chain controller. Useful during multi-processing.
                                    Defaults to 0. Not normally used by Algorithms.
        """
        self.log.info(
            "Finalize algorithm %s called at stage %d filenum %d",
            self.alg_name,
            stage,
            self.filenum,
        )
        # pylint:disable=pointless-string-statement

        # ---------------------------------------------------------------------
        # Add finalization steps here /
        # ---------------------------------------------------------------------

        """ None """

        # ---------------------------------------------------------------------
