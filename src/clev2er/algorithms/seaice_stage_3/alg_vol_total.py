"""clev2er.algorithms.seaice.alg_vol_total.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Calculates the totals for volume

#Main initialization (init() function) steps/resources required

None

#Main process() function steps

Calculate totals and add to shared_dict
Log totals to access them later

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

        """ None """

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
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

        """ Calculate totals and add to shared_dict
        Log totals to access them later """

        # total up values
        shared_dict["total_volume"] = np.sum(shared_dict["volume_grid"])
        shared_dict["total_fyi_volume"] = np.sum(
            shared_dict["volume_grid"] * shared_dict["frac_fyi_grid"]
        )
        shared_dict["total_myi_volume"] = np.sum(
            shared_dict["volume_grid"] * shared_dict["frac_myi_grid"]
        )

        shared_dict["total_area"] = np.sum(shared_dict["area_grid"])
        shared_dict["total_fyi_area"] = np.sum(
            shared_dict["area_grid"] * shared_dict["frac_fyi_grid"]
        )
        shared_dict["total_myi_area"] = np.sum(
            shared_dict["area_grid"] * shared_dict["frac_myi_grid"]
        )

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
