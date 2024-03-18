""" clev2er.algorithms.seaice.alg_sla_calculations.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose

    Calculates sea level anomaly (SLA) using elevation and MSS. Removes SLA values greater than 20m 
    or less than -20m. 

    #Main initialization (init() function) steps/resources required

    init_steps

    #Main process() function steps

    process_steps

    #Contribution to shared_dict

    contributions

    #Requires from shared_dict

    requirements

    Author: Ben Palmer
    Date: 12 Mar 2024
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

        # SLA related values
        self.track_limit = self.config["alg_sla_calculations"]["track_limit"]
        self.sample_limit = self.config["alg_sla_calculations"]["sample_limit"]

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

        sla = shared_dict["elevation"] - shared_dict["mss"]

        # find lead indices
        lead_indx = shared_dict["lead_floe_class"] == 2

        # find lead samples where sla is inside acceptable range
        indx_lead_sla_inside_range = np.isclose(sla[lead_indx], 0, atol=self.sample_limit)

        self.log.info(
            "Leads with SLA outside of range - %d", np.sum(np.invert(indx_lead_sla_inside_range))
        )

        # remove leads with SLAs outside of acceptable values
        sla[lead_indx][indx_lead_sla_inside_range] = np.nan

        # skip track if mean SLA of leads is outside of limit
        if not np.isclose(mean_sla := np.nanmean(sla[lead_indx]), 0, atol=self.track_limit):
            self.log.info("Mean SLA is outside of acceptable range - %d", mean_sla)
            self.log.info("Skipping file...")
            return (False, "SKIP_OK")

        self.log.info(
            "SLA - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f Count=%d NaN=%d",
            np.nanmean(sla),
            np.nanstd(sla),
            np.nanmin(sla),
            np.nanmax(sla),
            sla.shape[0],
            sum(np.isnan(sla)),
        )

        shared_dict["sea_level_anomaly"] = sla
        shared_dict["lead_indx"] = lead_indx
        shared_dict["indx_lead_sla_inside_range"] = indx_lead_sla_inside_range

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
