"""clev2er.algorithms.seaice.alg_vol_fill_nn.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Uses a nearest neighbour search to fill in missing volume measurements

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

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import Geod

from clev2er.algorithms.base.base_alg import BaseAlgorithm


class Algorithm(BaseAlgorithm):
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

        self.nlats = self.config["shared"]["grid_nlats"]
        self.nlons = self.config["shared"]["grid_nlons"]
        self.nn_radius = self.config["alg_vol_fill_nn"]["nn_radius"]
        self.projection = self.config["alg_vol_fill_nn"]["working_projection"]

        self.geod = Geod(ellps=self.projection)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
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

        # ==========================================================================================
        # Compute nearest neighbour interpolation for empty cells
        # ==========================================================================================
        self.log.info("Computing nearest neighbour interpolation...")

        # We're finding the number of lats to search around the target cell
        for ilat in range(self.nlats + 1):
            lat = 40 + (ilat * 0.1)
            _, _, dist = self.geod.inv(-180, lat, -180, 40)
            if dist > self.nn_radius:
                break
            if ilat == self.nlats:
                return (False, "ERROR_NN_SEARCH_RADIUS")

        search_lat_range = ilat - 1

        # this part is O(n^4), there must be a better way to do this?
        # scipy.KDTree or sklearn.NearestNeighbor probably
        # See notes for plans for optimisation
        for ilat in range(self.nlats):
            # finding the number of lons to seach for around the target cell
            for ilon in range(self.nlons + 1):
                lon = -180 + (ilon * 0.5)
                _, _, dist = self.geod.inv(lon, ilat, -180, 40)
                if dist > self.nn_radius:
                    break
                if ilat == self.nlats:
                    return (False, "ERROR_NN_SEARCH_RADIUS")

            search_lon_range = ilon - 1

            for ilon in range(self.nlons):
                # specifying the search area
                search_lat_min = max(ilat - search_lat_range, 0)
                search_lat_max = min(ilat + search_lat_range, self.nlats - 1)
                search_lon_min = max(ilon - search_lon_range, 0)
                search_lon_max = min(ilon + search_lon_range, self.nlons - 1)

                distmin = self.nn_radius

                # finds the closest cell and copies the values over
                for iilat in range(search_lat_min, search_lat_max):
                    for iilon in range(search_lon_min, search_lon_max):
                        # need to convert indexes to lon lats here
                        lat_1 = 40 + (ilat * 0.1)
                        lon_1 = -180 + (ilon * 0.5)
                        lat_2 = 40 + (iilat * 0.1)
                        lon_2 = -180 + (iilon * 0.5)
                        _, _, dist = self.geod.inv(lon_1, lat_1, lon_2, lat_2)
                        if dist >= distmin:
                            continue
                        distmin = dist
                        fill_thk[ilat, ilon] = shared_dict["thickness_grid"][iilat, iilon]
                        fill_conc[ilat, ilon] = shared_dict["iceconc_grid"][iilat, iilon]
                        fill_vol[ilat, ilon] = shared_dict["volume_grid"][iilat, iilon]
                        fill_fyi[ilat, ilon] = shared_dict["frac_fyi_grid"][iilat, iilon]
                        fill_myi[ilat, ilon] = shared_dict["frac_myi_grid"][iilat, iilon]
                        fill_nin[ilat, ilon] = shared_dict["number_in"][iilat, iilon]

        # fill in empty cells with nearest neighbour where possible
        for ilat in range(self.nlats):
            for ilon in range(self.nlons):
                if fill_nin[ilat, ilon] > 0:
                    # check we're not trying to fill in a cell which isnt empty
                    if shared_dict["number_in"][ilat, ilon] > 0:
                        return (False, "FILLING_NONEMPTY_CELL")

                    # copy values over
                    shared_dict["thickness_grid"][ilat, ilon] = fill_thk[ilat, ilon]
                    shared_dict["iceconc_grid"][ilat, ilon] = fill_conc[ilat, ilon]
                    shared_dict["volume_grid"][ilat, ilon] = fill_vol[ilat, ilon]
                    shared_dict["frac_fyi_grid"][ilat, ilon] = fill_fyi[ilat, ilon]
                    shared_dict["frac_myi_grid"][ilat, ilon] = fill_myi[ilat, ilon]
                    shared_dict["number_in"][ilat, ilon] = fill_nin[ilat, ilon]
                    if shared_dict["number_in"][ilat, ilon] > 0:
                        # not sure what this does, pretty sure we check for this several lines above
                        # might change from fill value but why?
                        shared_dict["gaps"][ilat, ilon] = 0
                        shared_dict["area_grid"][ilat, ilon] = 1.0

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
