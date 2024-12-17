"""clev2er.algorithms.seaice_stage_1.alg_threshold_retrack.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Algorithm for retracking the smoothed diffuse waveforms which represent ice floes in sea
ice processing. Retracks waveform to X% of the waveform, then Y% of the waveform, where X < Y.
We measure the bin distance between the retracking values, and any sample which has a distance
greater than the limit will not be processed further.

#Main initialization (init() function) steps/resources required

Get threshold and leading edge width settings from config file

#Main process() function steps

Retrack to lower threshold point
Retrack to higher threshold point
Generate index of which samples have a LEW less than the limit

#Contribution to shared_dict

shared_dict["floe_retracking_points"] (np.ndarray[float]) : array of retracking points for floes
shared_dict["idx_lew_gt_max"] (np.ndarray[int]) :   index array of samples with leading edge
                                                    width greater than limit

#Requires from shared_dict

shared_dict["waveform_smooth"]

Author: Ben Palmer
Date: 23 Feb 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.cs2.retrackers.threshold_retracker import threshold_retracker


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

        self.threshold_high = self.config["alg_threshold_retrack"]["threshold_high"]
        self.threshold_low = self.config["alg_threshold_retrack"]["threshold_low"]

        self.lew_max = self.config["alg_threshold_retrack"]["lew_max"]

        if self.threshold_high <= self.threshold_low:
            raise ValueError(
                "Threshold config values for alg_threshold_retrack are in the wrong order."
            )

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

        if shared_dict["diffuse_index"].size == 0:
            self.log.info("No diffuse waves in file - skipping retracking...")
            shared_dict["floe_retracking_points"] = np.array([])
            return (success, error_str)

        points_higher = np.apply_along_axis(
            threshold_retracker, 1, shared_dict["waveform_smooth"], threshold=self.threshold_high
        )

        points_lower = np.apply_along_axis(
            threshold_retracker, 1, shared_dict["waveform_smooth"], threshold=self.threshold_low
        )

        lews = np.abs(points_higher - points_lower)

        # only keep points with leading edge width < max value
        idx_lew_gt_max = np.where(lews > self.lew_max)[0]

        self.log.info(
            "Number of samples that exceeded LEW limit - %d",
            shared_dict["waveform_smooth"].shape[0] - idx_lew_gt_max.size,
        )

        shared_dict["floe_retracking_points"] = points_higher
        shared_dict["idx_lew_gt_max"] = idx_lew_gt_max

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

        # ---------------------------------------------------------------------
