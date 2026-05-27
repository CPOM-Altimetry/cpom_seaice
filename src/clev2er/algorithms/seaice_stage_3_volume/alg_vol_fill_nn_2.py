"""clev2er.algorithms.seaice.alg_vol_fill_nn.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Uses a nearest neighbour search to fill in missing volume measurements

NOTE: This version is experimental, and uses nearest neighbour trees to find the nearest
grid cells to fill from. This version doesn't work exactly how we want it yet, and needs
further work to improve it.

#Main initialization (init() function) steps/resources required

Read config options

#Main process() function steps

Create empty arrays for grids used in filling process
Compute nearest neighbours for each cell
Fill in empty cells using nearest neighbours

#Main finalize() function steps

None

#Contribution to shared_dict

contributions

#Requires from shared_dict

requirements

Author: Ben Palmer
Date: 06 Jan 2025
"""

import warnings
from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import CRS
from sklearn.neighbors import BallTree

from clev2er.algorithms.base.base_alg import BaseAlgorithm

warnings.filterwarnings("error")


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
        # pylint: disable=pointless-string-statement
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
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ Read config options """

        self.nlats = self.config["shared"]["nlats"]
        self.nlons = self.config["shared"]["nlons"]
        self.ncells = self.nlats * self.nlons
        self.nn_radius = self.config["alg_vol_fill_nn"]["nn_radius"]
        self.projection = self.config["alg_vol_fill_nn"]["working_projection"]

        self.earth_radius = 6356.752

        self.geod = CRS(self.projection).get_geod()

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
        # pylint: disable=unbalanced-tuple-unpacking
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
            Create empty arrays for grids used in filling process
            Compute nearest neighbours for each cell
            Fill in empty cells using nearest neighbours
        """

        fill_thk = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_conc = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_vol = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_fyi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_myi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_nin = np.zeros((self.nlats, self.nlons), dtype=np.float64)

        # fill statistics
        fill_flag = np.zeros((self.nlats, self.nlons), dtype=np.int32)
        n_cells_filled = 0

        # ==========================================================================================
        # Compute nearest neighbour interpolation for empty cells
        # ==========================================================================================
        self.log.info("Computing nearest neighbour interpolation...")

        lats = np.arange(0, self.nlats)
        lons = np.arange(0, self.nlons)

        lon_mesh, lat_mesh = np.meshgrid(lons, lats)

        lon_mesh = -180 + lon_mesh * 0.5
        lat_mesh = 40 + lat_mesh * 0.1

        all_points = np.transpose([lat_mesh.flatten(), lon_mesh.flatten()]) * np.pi / 180

        valid_fill_values = shared_dict["number_in"].flatten() > 0
        tree_where = np.where(valid_fill_values)[0]
        tree_points = all_points[valid_fill_values]

        file_point_tree = BallTree(tree_points, metric="haversine")
        neighbour_distances, file_neighbouring_indices = file_point_tree.query_radius(
            all_points,
            self.nn_radius / self.earth_radius,
            return_distance=True,
        )
        neighbour_distances = neighbour_distances * self.earth_radius  # convert to km

        for index, (distances, neighbours) in enumerate(
            zip(neighbour_distances, file_neighbouring_indices)
        ):
            if len(distances) == 0:
                continue

            ilat, ilon = np.unravel_index(index, (self.nlats, self.nlons))
            if shared_dict["number_in"][ilat, ilon] > 0:
                continue

            closest_neighbour = np.argmin(distances)

            if distances[closest_neighbour] <= self.nn_radius:
                closest_point_index = tree_where[neighbours[closest_neighbour]]

                iilat, iilon = np.unravel_index(closest_point_index, (self.nlats, self.nlons))

                if ilat == 289 and ilat == 448:
                    print(index, closest_point_index)
                    print(
                        distances[closest_neighbour],
                        (ilat, ilon),
                        (iilat, iilon),
                        shared_dict["number_in"][iilat, iilon],
                    )

                fill_thk[ilat, ilon] = shared_dict["thickness_grid"][iilat, iilon]
                fill_conc[ilat, ilon] = shared_dict["iceconc_grid"][iilat, iilon]
                fill_vol[ilat, ilon] = shared_dict["volume_grid"][iilat, iilon]
                fill_fyi[ilat, ilon] = shared_dict["frac_fyi_grid"][iilat, iilon]
                fill_myi[ilat, ilon] = shared_dict["frac_myi_grid"][iilat, iilon]
                fill_nin[ilat, ilon] = shared_dict["number_in"][iilat, iilon]

        # ==============================================================================
        # fill in empty cells with nearest neighbour where possible
        # ==============================================================================
        self.log.info("Filling in empty cells...")
        for ilat in np.arange(self.nlats):
            # skip row if no useful values exist
            if (fill_nin[ilat, :] == 0).all():
                continue

            for ilon in np.arange(self.nlons):
                if fill_nin[ilat, ilon] > 0:
                    # check we're not trying to fill in a cell which isnt empty
                    if shared_dict["number_in"][ilat, ilon] > 0:
                        return (False, "FILLING_NONEMPTY_CELL")

                    # mark cell as filled
                    fill_flag[ilat, ilon] = 1
                    n_cells_filled += 1

                    # copy values over
                    shared_dict["thickness_grid"][ilat, ilon] = fill_thk[ilat, ilon]
                    shared_dict["iceconc_grid"][ilat, ilon] = fill_conc[ilat, ilon]
                    shared_dict["volume_grid"][ilat, ilon] = fill_vol[ilat, ilon]
                    shared_dict["frac_fyi_grid"][ilat, ilon] = fill_fyi[ilat, ilon]
                    shared_dict["frac_myi_grid"][ilat, ilon] = fill_myi[ilat, ilon]
                    shared_dict["number_in"][ilat, ilon] = fill_nin[ilat, ilon]

                    # not sure what this does, pretty sure we check for this several lines above
                    # might change from fill value but why?
                    # might be redundant now as we constantly check if the value we're filling with
                    # is >0 anyways
                    if shared_dict["number_in"][ilat, ilon] > 0:
                        shared_dict["gaps"][ilat, ilon] = 0
                        shared_dict["area_grid"][ilat, ilon] = 1.0

        self.log.info("Number of cells filled: %d", n_cells_filled)

        # Add filling grids to shared_mem
        shared_dict["fill_thick"] = fill_thk
        shared_dict["fill_nin"] = fill_nin
        shared_dict["fill_flag"] = fill_flag

        # -------------------------------------------------------------------
        # Returns (True,'') if success

        return (success, error_str)

    def finalize(self, stage: int = 0) -> None:
        # pylint: disable=pointless-string-statement
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
        # ---------------------------------------------------------------------
        # Add finalization steps here /
        # ---------------------------------------------------------------------

        """ None """

        # ---------------------------------------------------------------------
