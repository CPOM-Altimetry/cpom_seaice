""" clev2er.algorithms.seaice.alg_ice_class.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose

    Assigns the class of samples in the file

    #Main initialization (init() function) steps/resources required

    Get ice concentration threshold from config

    #Main process() function steps

    Create an array of 0s (default values)
    Set specular echoes to 2 (lead class)
    Set all diffuse echoes to 1 (ocean class)
    Set diffuse echoes with ice concentration greater than threshold to 3 (floe class)

    #Contribution to shared_dict

    lead_floe_class (np.ndarray[int]) : Class of whether each waveform is a lead, floe,
                                        ocean or unclassified

    #Requires from shared_dict

    specular_index 
    diffuse_echoes 
    seaice_concentration

    Author: Ben Palmer
    Date: 08 Mar 2024
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

        self.conc_threshold = self.config["alg_ice_class"]["seaice_concentration_threshold"]

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

        # make surface type class
        # specular echoes = leads = 1
        # diffuse echoes = floes or oceans = 2 or 3
        # floes = diffuse echoes w/ ice conc > 75.0%
        # oceans = diffuse echoes w/ ice conc < 75.0%

        shared_dict["lead_floe_class"] = np.zeros(shared_dict["sat_lat"].shape[0], dtype=int)

        # specular waves can only be leads
        shared_dict["lead_floe_class"][shared_dict["specular_index"]] = 2

        # diffuse waves can be oceans or floes, set to ocean first
        shared_dict["lead_floe_class"][shared_dict["diffuse_index"]] = 1

        # find diffuse waves which have conc > threshold, set to floes
        shared_dict["lead_floe_class"][
            shared_dict["diffuse_index"][
                shared_dict["seaice_concentration"][shared_dict["diffuse_index"]]
                > self.conc_threshold
            ]
        ] = 3

        self.log.info("Class counts")
        for v in set([0, 1, 2, 3]).union(np.unique(shared_dict["lead_floe_class"])):
            self.log.info("\t %d - %5d", v, sum(shared_dict["lead_floe_class"] == v))

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
