"""clev2er.algorithms.seaice_stage_1.alg_smooth_waveform.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Algorithm for smoothing diffuse waveforms identified in previous steps

#Main initialization (init() function) steps/resources required

Get number of points to use in moving average from config

#Main process() function steps

Get diffuse waves using index
Smooth waveforms using smoothing function
Store waveforms in shared_dict

#Contribution to shared_dict

shared_dict["waveform_smoothed"]

#Requires from shared_dict

shared_dict["diffuse_index"]
shared_dict["waveform"]

Author: Ben Palmer
Date: 23 Feb 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.waveforms.smooth import smooth_waveform


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

        self.moving_average_width = self.config["alg_smooth_waveform"]["moving_average_width"]
        if self.moving_average_width % 2 == 0:
            raise ValueError("Moving average width must be an odd integer.")

        self.log.info("     Number of points in moving average - %d", self.moving_average_width)

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
            self.log.info("No diffuse waves in file - skipping smoothing...")
            shared_dict["waveform_smooth"] = np.array([])
            return (success, error_str)

        diffuse_waves = shared_dict["waveform"][shared_dict["diffuse_index"]]

        smoothed_waves = np.apply_along_axis(
            smooth_waveform, 1, diffuse_waves, moving_avg_size=self.moving_average_width
        )

        shared_dict["waveform_smooth"] = np.asarray(smoothed_waves)

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
