""" clev2er.algorithms.seaice.waveform_discrimination.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose
    
    Disciminates whether each waveform echoe is diffuse or specular from the pulse peakiness and 
    stack standard deviation.
    
    #Main initialization (init() function) steps/resources required

    alg_wave_discrimination:
        diffuse_peakiness(float): peakiness threshold for diffuse waves
        specular_peakiness(float): peakiness threshold for specular waves
        sar_ssd(float): ssd threshold for waves found during SAR operating mode (SAR file)
        sin_ssd(float): ssd threshold for waves found during SARIn operating mode (SIN file)

    #Main process() function steps
    
    Choose ssd threshold based on operating mode
    Find index of diffuse waves
    Find index of specular waves
    Create lead floe class
    Set lead floe class of specular waves to leads
    Set lead floe class of diffuse waves to ocean (floes found later)
    
    #Contribution to shared_dict

    shared_dict["specular_index"] (np.array[int]) : index of specular waves 
                                                    in shared_dict["waveform"]
    shared_dict["diffuse_index"] (np.array[int]) : index of diffuse waves in shared_dict["waveform"]
    shared_dict["lead_floe_class"] (np.array[int]) : Class of whether each waveform is a lead, floe,
                                                    ocean or unclassified
    
    #Requires from shared_dict

    shared_dict["waveform"]
    shared_dict["waveform_ssd"]
    shared_dict["pulse_peakiness"]
    shared_dict["instr_mode"]
    
"""

from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

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

        self.diffuse_peakiness = self.config["alg_wave_discrimination"]["diffuse_peakiness"]
        self.specular_peakiness = self.config["alg_wave_discrimination"]["specular_peakiness"]
        self.sar_ssd = self.config["alg_wave_discrimination"]["sar_ssd"]
        self.sin_ssd = self.config["alg_wave_discrimination"]["sin_ssd"]

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

        # different SSD threshold depending on the operating mode
        if shared_dict["instr_mode"] == "SAR":
            ssd_threshold = self.sar_ssd
        elif shared_dict["instr_mode"] == "SIN":
            ssd_threshold = self.sin_ssd

        # find diffuse waveforms

        diffuse_waves = (shared_dict["pulse_peakiness"] < self.diffuse_peakiness) & (
            shared_dict["waveform_ssd"] > ssd_threshold
        )

        self.log.info("Number of diffuse waves - %d", sum(diffuse_waves))

        # find specular waveforms

        specular_waves = (shared_dict["pulse_peakiness"] > self.specular_peakiness) & (
            shared_dict["waveform_ssd"] < ssd_threshold
        )

        self.log.info("Number of specular waves - %d", sum(specular_waves))

        # make indexes for each
        shared_dict["specular_index"] = np.where(specular_waves)[0]
        shared_dict["diffuse_index"] = np.where(diffuse_waves)[0]

        # make surface type class
        # specular echoes = leads = 1
        # diffuse echoes = floes or oceans = 2 or 3
        # (will set diffuse waves to oceans and work out which are floes later)

        shared_dict["lead_floe_class"] = np.zeros(shared_dict["sat_lat"].shape[0], dtype=int)

        shared_dict["lead_floe_class"][shared_dict["specular_index"]] = 1
        shared_dict["lead_floe_class"][shared_dict["diffuse_index"]] = 3

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
