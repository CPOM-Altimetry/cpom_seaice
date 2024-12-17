"""clev2er.algorithms.seaice_stage_1.alg_elev_calculations.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Calculates the elevation for each sample, adjusting elevations from diffuse samples with the
retracker bias.

#Main initialization (init() function) steps/resources required

Set the satellite bin width and speed of light variables

#Main process() function steps

Calculate the total geophysical corrections by adding all corrections together.
Calculate the range of the satellite to the surface using the window delay.
Calculate the retracking correction.
Calculate the elevation subtracting the satellite range, geophysical corrections, and
    retracking correction from the satellite altitude.
Subtract the retracker bias from the elevations from diffuse waveforms.
Save elevations to dict.

#Contribution to shared_dict

'elevation' (np.array[float]) : the elevation of the surface above the WGS84 reference ellipsoid

#Requires from shared_dict

'window_delay'
'sat_altitude'
'wet_trop_correction'
'dry_trop_correction'
'inv_baro_correction'
'iono_correction'
'ocean_tide'
'long_period_tide'
'loading_tide'
'earth_tide'
'pole_tide'
'specular_index'
'diffuse_index'
'idx_lew_gt_max'
'lead_retracking_points'
'floe_retracking_points'
'bin_shift'

Author: Ben Palmer
Date: 05 Mar 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

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

        self.c = self.config["alg_elev_calculations"]["speed_of_light"]
        self.bin_width = self.config["alg_elev_calculations"]["bin_width"]
        self.tracking_bin = self.config["alg_elev_calculations"]["tracking_bin"]
        self.diffuse_retracker_bias = self.config["alg_elev_calculations"]["diffuse_retracker_bias"]

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
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
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        geophysical_corrections = (
            shared_dict["wet_trop_correction"]
            + shared_dict["dry_trop_correction"]
            + shared_dict["inv_baro_correction"]
            + shared_dict["iono_correction"]
            + shared_dict["ocean_tide"]
            + shared_dict["long_period_tide"]
            + shared_dict["loading_tide"]
            + shared_dict["earth_tide"]
            + shared_dict["pole_tide"]
        )

        sat_range = self.c / 2 * shared_dict["window_delay"]

        retracking_points = np.zeros(shared_dict["sat_lat"].size) * np.nan
        retracking_points[shared_dict["specular_index"]] = shared_dict["lead_retracking_points"]
        retracking_points[shared_dict["diffuse_index"]] = shared_dict["floe_retracking_points"]

        # do bin_width * bin_shift here since it will all be subtracted from the elevation
        retracker_correction = self.bin_width * (
            retracking_points - self.tracking_bin - shared_dict["bin_shift"]
        )

        elevations = shared_dict["sat_altitude"] - (
            sat_range + retracker_correction + geophysical_corrections
        )

        # Add retracker bias from the diffuse retracker
        elevations[shared_dict["diffuse_index"]] -= self.diffuse_retracker_bias

        self.log.info(
            "Elevation - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f Count=%d NaNs=%d",
            np.nanmean(elevations),
            np.nanstd(elevations),
            np.nanmin(elevations),
            np.nanmax(elevations),
            elevations.shape[0],
            sum(np.isnan(elevations)),
        )
        shared_dict["elevation"] = elevations

        shared_dict["valid"][shared_dict["diffuse_index"][shared_dict["idx_lew_gt_max"]]] = False
        shared_dict["valid"][shared_dict["diffuse_index"][shared_dict["idx_lew_gt_max"]]] = False

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
        # Add finalization steps here \/
        # ---------------------------------------------------------------------

        # None

        # ---------------------------------------------------------------------
