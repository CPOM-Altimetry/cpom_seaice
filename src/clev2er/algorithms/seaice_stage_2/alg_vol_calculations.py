"""clev2er.algorithms.seaice.alg_vol_calculations.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Calculates ice volume from ice thickness values and auxilliary data

#Main initialization (init() function) steps/resources required

Read params from config

#Main process() function steps

Create numpy arrays for thickness, conc, volume, thick_fyi, thick_myi, fraction of fyi and myi,
counts, and fill values
Loop through values and get index of array locations
Add thickness and concentration values from samples to arrays, increase count by 1 each time
Do the same with fyi and myi thickness
Compute nearest neighbour interpolation for empty cells
Fill empty cells with nearest neighbour where possible
Apply ocean fraction values
Apply cell area values
Apply ice concentration mask
Apply region mask
Total up volume and area

#Main finalize() function steps

None

#Contribution to shared_dict

volume_grid
thickness_grid
conc_grid
thick_fyi_grid
thick_myi_grid
frac_fyi_grid
frac_myi_grid,
count_grid,
count_fyi_grid,
count_myi_grid,
gaps_grid

#Requires from shared_dict

requirements

Author: Ben Palmer
Date: 19 Sep 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import Geod

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# pylint:disable=pointless-string-statement


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

        """ Read params from config """
        self.nlats = self.config["alg_vol_calculations"]["nlats"]
        self.nlons = self.config["alg_vol_calculations"]["nlons"]
        self.ninmin = self.config["alg_vol_calculations"]["ninmin"]
        self.nn_radius = self.config["alg_vol_calculations"]["nn_radius"]
        self.projection = self.config["shared"]["output_projection"]

        self.geod = Geod(ellps=self.projection)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
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
        # pylint:disable=too-many-statements
        # pylint:disable=too-many-branches

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # /    down the chain in the 'shared_dict' dict     /
        # -------------------------------------------------------------------

        """ Create numpy arrays for thickness, conc, volume, thick_fyi, thick_myi, 
            fraction of fyi and myi, counts, and fill values
            Loop through values and get index of array locations
            Add thickness and concentration values from samples to arrays, 
                increase count by 1 each time
            Do the same with fyi and myi thickness
            Compute nearest neighbour interpolation for empty cells
            Fill empty cells with nearest neighbour where possible
            Apply ocean fraction values
            Apply cell area values
            Apply ice concentration mask
            Apply region mask
            Total up volume and area 
        """

        # initialise all the arrays we need
        thickness = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        iceconc = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        volume = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_thk = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_vol = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_fyi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_myi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        fill_conc = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        frac_fyi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        frac_myi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        thick_fyi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        thick_myi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        area = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        gaps = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        count = np.zeros((self.nlats, self.nlons), dtype=np.int64)
        fill_nin = np.zeros((self.nlats, self.nlons), dtype=np.int64)
        nin_fyi = np.zeros((self.nlats, self.nlons), dtype=np.int64)
        nin_myi = np.zeros((self.nlats, self.nlons), dtype=np.int64)
        nin = np.zeros((self.nlats, self.nlons), dtype=np.int64)

        # calculate the indexes of each record
        ilats = ((l1b["sat_lats"][:].data - 40) / 0.1).astype(int)
        ilons = ((l1b["sat_lons"][:].data + 180) / 0.5).astype(int)

        # add thickness data to array with indexes
        thickness[ilats, ilons] += shared_dict["thickness"]
        iceconc[ilats, ilons] += l1b["iceconc"][:].data
        nin[ilats, ilons] += 1

        sample_fyi = shared_dict["seaice_type"] == 2
        sample_myi = shared_dict["seaice_type"] == 3

        # calculate values for mfy and fyi
        thick_fyi[ilats, ilons][sample_fyi] += shared_dict["thickness"]
        nin_fyi[ilats, ilons][sample_fyi] += 1

        thick_myi[ilats, ilons][sample_myi] += shared_dict["thickness"]
        nin_myi[ilats, ilons][sample_myi] += 1

        # calculate thickness, conc and volume
        for ilat in range(self.nlats):
            for ilon in range(self.nlons):
                if nin[ilat, ilon] > self.ninmin:  # this prevents divide by 0 error
                    thickness[ilat, ilon] /= nin[ilat, ilon]
                    iceconc[ilat, ilon] /= nin[ilat, ilon]
                    volume[ilat, ilon] = thickness[ilat, ilon] * 0.001 * iceconc[ilat, ilon] * 0.01

                # again stopping divide by 0 error
                if thick_fyi[ilat, ilon] > 0 or thick_myi[ilat, ilon] > 0:
                    frac_fyi[ilat, ilon] = thick_fyi[ilat, ilon] / (
                        thick_fyi[ilat, ilon] + thick_myi[ilat, ilon]
                    )
                    frac_myi[ilat, ilon] = thick_myi[ilat, ilon] / (
                        thick_fyi[ilat, ilon] + thick_myi[ilat, ilon]
                    )

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

        search_lat = ilat - 1

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

            search_lon = ilon - 1

            for ilon in range(self.nlons):
                # specifying the search area
                ilatmin = (ilat - search_lat) if (ilat - search_lat) >= 0 else 0
                ilatmax = (
                    (ilat + search_lat) if (ilat + search_lat) < self.nlats else self.nlats - 1
                )
                ilonmin = (ilon - search_lon) if (ilon - search_lon) >= 0 else 0
                ilonmax = (
                    (ilon + search_lon) if (ilon + search_lon) < self.nlons else self.nlons - 1
                )

                distmin = self.nn_radius

                # finds the closest cell and copies the values over
                for iilat in range(ilatmin, ilatmax):
                    for iilon in range(ilonmin, ilonmax):
                        _, _, dist = self.geod.inv(ilon, ilat, iilon, iilat)
                        if dist >= distmin:
                            continue
                        fill_thk[ilat, ilon] = thickness[iilat, iilon]
                        fill_conc[ilat, ilon] = iceconc[iilat, iilon]
                        fill_vol[ilat, ilon] = volume[iilat, iilon]
                        fill_fyi[ilat, ilon] = frac_fyi[iilat, iilon]
                        fill_myi[ilat, ilon] = frac_myi[iilat, iilon]
                        fill_nin[ilat, ilon] = nin[iilat, iilon]

        # fill in empty cells with nearest neighbour where possible
        for ilat in range(self.nlats):
            for ilon in range(self.nlons):
                if fill_nin[ilat, ilon] > 0:
                    # check we're not trying to fill in a cell which isnt empty
                    if nin[ilat, ilon] > 0:
                        return (False, "FILLING_NONEMPTY_CELL")

                    # copy values over
                    thickness[ilat, ilon] = fill_thk[ilat, ilon]
                    iceconc[ilat, ilon] = fill_conc[ilat, ilon]
                    volume[ilat, ilon] = fill_vol[ilat, ilon]
                    frac_fyi[ilat, ilon] = fill_fyi[ilat, ilon]
                    frac_myi[ilat, ilon] = fill_myi[ilat, ilon]
                    nin[ilat, ilon] = fill_nin[ilat, ilon]
                    if nin[ilat, ilon] > 0:
                        # not sure what this does, pretty sure we check for this several lines above
                        # might change from fill value but why?
                        gaps[ilat, ilon] = 0
                        area[ilat, ilon] = 1.0

        # apply aux values
        # ocean fraction
        volume *= shared_dict["ocean_frac"]
        area *= shared_dict["ocean_frac"]

        # cell area
        volume *= shared_dict["cell_area"]
        area *= shared_dict["cell_area"]

        # extent mask
        thickness *= shared_dict["extent_mask"]
        volume *= shared_dict["extent_mask"]
        iceconc *= shared_dict["extent_mask"]
        area *= shared_dict["extent_mask"]

        # region mask
        thickness *= shared_dict["region_mask"]
        volume *= shared_dict["region_mask"]
        iceconc *= shared_dict["region_mask"]
        area *= shared_dict["region_mask"]

        # total up values
        shared_dict["total_volume"] = np.sum(volume)
        shared_dict["total_fyi_volume"] = np.sum(volume * frac_fyi)
        shared_dict["total_myi_volume"] = np.sum(volume * frac_myi)

        shared_dict["total_area"] = np.sum(area)
        shared_dict["total_fyi_area"] = np.sum(area * frac_fyi)
        shared_dict["total_myi_area"] = np.sum(area * frac_myi)

        # lots of logging :)
        self.log.info("- Volume calculations -")
        self.log.info("    All")
        self.log.info("        Total Volume -\t%f", shared_dict["total_volume"])
        self.log.info("        Total Area -\t%f", shared_dict["total_area"])
        self.log.info("    FYI")
        self.log.info("        Total Volume -\t%f", shared_dict["total_fyi_volume"])
        self.log.info("        Total Area -\t%f", shared_dict["total_fyi_area"])
        self.log.info("    MYI")
        self.log.info("        Total Volume -\t%f", shared_dict["total_myi_volume"])
        self.log.info("        Total Area -\t%f", shared_dict["total_myi_area"])

        # add arrays to shared_dict
        shared_dict["volume_grid"] = volume
        shared_dict["thickness_grid"] = thickness
        shared_dict["conc_grid"] = iceconc
        shared_dict["thick_fyi_grid"] = thick_fyi
        shared_dict["thick_myi_grid"] = thick_myi
        shared_dict["frac_fyi_grid"] = frac_fyi
        shared_dict["frac_myi_grid"] = frac_myi
        shared_dict["count_grid"] = count
        shared_dict["area_grid"] = area
        shared_dict["nin_grid"] = nin

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
        # ---------------------------------------------------------------------
        # Add finalization steps here /
        # ---------------------------------------------------------------------

        """ finalize """

        # ---------------------------------------------------------------------
