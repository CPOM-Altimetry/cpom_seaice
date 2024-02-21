""" clev2er.algorithms.seaice.alg_pulse_peakiness.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose
    
    Calculates the pulse peakiness of each waveform, which is used later to discriminate
    between specular and diffuse waveforms. Described in https://doi.org/10.1016/j.asr.2017.10.051
    
    #Main initialization (init() function) steps/resources required

    None

    #Main process() function steps
    
    Define peakiness calculation
    Apply to every waveform
    Add to shared_dict
    
    #Contribution to shared_dict

    shared_dict["pulse_peakiness"] (np.array[float]) : the pulse peakiness of each waveform

    #Requires from shared_dict

    shared_dict["waveform"]
    
"""

from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.waveforms.peakiness import peakiness

# each algorithm shares some common class code, so pylint: disable=duplicate-code


class Algorithm(BaseAlgorithm):
    """CLEV2ER Algorithm class

    contains:
         .log (Logger) : log instance that must be used for all logging, set by BaseAlgorithm
         .config (dict) : configuration dictionary, set by BaseAlgorithm
         - functions that need completing:
         .init() : Algorithm initialization function (run once at start of chain)
         .process(l1b,shared_dict) : Algorithm processing function (run on every L1b file)
         .finalize() : Algorithm finalization/closure function (run after all chain
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

        self.noise_floor_start = self.config["alg_pulse_peakiness"]["noise_floor_start_bin"]
        self.noise_floor_end = self.config["alg_pulse_peakiness"]["noise_floor_end_bin"]

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

        pp = np.apply_along_axis(
            peakiness,
            1,
            shared_dict["waveform"],
            noise_floor_start=self.noise_floor_start,
            noise_floor_end=self.noise_floor_end + 1,
        )

        self.log.info("Number of peakiness values - %d", pp.shape[0])
        self.log.info("Number of NaN values returned - %d", sum(np.isnan(pp)))

        shared_dict["pulse_peakiness"] = pp

        # -------------------------------------------------------------------
        # Returns (True,'') if success
        return (success, error_str)

    def finalize(self, stage: int = 0) -> None:
        """Algorithm finalization function - called after all processing completed

          Can be used to clean up/free resources initialized in the init() function

        Args:
            stage (int, optional): this sets the stage when this function is called
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

        # none required in this algorithm

        # ---------------------------------------------------------------------
