"""clev2er.algorithms.seaice_stage_1.alg_giles_retrack.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Retracks specular waves using the gaussian plus exponential retracker, or Giles' retracker

#Main initialization (init() function) steps/resources required

If max_iterations has been set, use that value, else use the default value

#Main process() function steps

Get specular waves from waveform using specular_index
Get retracking points using retracker function
Save specular retracking points as lead_retracking_points

#Contribution to shared_dict

'lead_retracking_points' (npt.NDarray) : Retracking points found using this algorithm

#Requires from shared_dict

'waveform'
'specular_index'

Author: Ben Palmer
Date: 01 Mar 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.cs2.retrackers.gaussian_exponential_retracker import (
    gauss_plus_exp_tracker,
)


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

        self.max_iterations = self.config["alg_giles_retrack"]["max_iterations"]
        self.max_fit_err = self.config["alg_giles_retrack"]["max_fit_err"]
        self.max_fit_sigma = self.config["alg_giles_retrack"]["max_fit_sigma"]

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

        if shared_dict["specular_index"].size == 0:
            self.log.info("No specular waves in file - skipping retracking...")
            shared_dict["lead_retracking_points"] = np.asarray([])
            return (success, error_str)

        specular_waves = shared_dict["waveform"][shared_dict["specular_index"]]

        retracker_output = np.apply_along_axis(
            gauss_plus_exp_tracker, 1, specular_waves, max_iterations=self.max_iterations
        )
        lead_retracking_points = retracker_output[:, 0]
        fit_qualities = retracker_output[:, 1]
        fit_sigmas = retracker_output[:, 2]

        bad_fits = (
            (0 > fit_qualities)
            | (self.max_fit_err < fit_qualities)
            | (fit_sigmas > self.max_fit_sigma)
            | (fit_sigmas < 0.00001)
        )

        lead_retracking_points[bad_fits] = np.nan

        num_nans = np.sum(np.isnan(lead_retracking_points))

        self.log.info("Number of NaN values returned by retracker - %d", num_nans)

        # If all retracking points are nans, skip file
        if num_nans == lead_retracking_points.size:
            return (False, "SKIP_OK")

        shared_dict["lead_retracking_points"] = lead_retracking_points

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

        # None needed

        # ---------------------------------------------------------------------
