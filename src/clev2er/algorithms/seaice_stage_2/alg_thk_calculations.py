"""clev2er.algorithms.seaice.alg_thk_calculations.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Calculates the thickness of each ice sample from the freeboard

#Main initialization (init() function) steps/resources required

Load in parameters from config

#Main process() function steps

Compute thickness with snow depth


#Main finalize() function steps

None

#Contribution to shared_dict

thickness: np.ndarray[np.float32] = ice thickness of each sample

#Requires from shared_dict

freeboard
seaice_type
snow_depth
snow_density

Author: Ben Palmer
Date: 05 Sep 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# pylint: disable=pointless-string-statement


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

        """ Load in parameters from config """
        self.rho_fyi = self.config["alg_thk_calculations"]["rho_fyi"]
        self.rho_myi = self.config["alg_thk_calculations"]["rho_myi"]
        self.rho_sea = self.config["alg_thk_calculations"]["rho_sea"]

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

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # /    down the chain in the 'shared_dict' dict     /
        # -------------------------------------------------------------------

        """ Compute thickness with snow depth """

        ice_densities = np.zeros(l1b["measurement_time"][:].size) * np.nan
        ice_densities[shared_dict["seaice_type"] == 2] = self.rho_fyi
        ice_densities[shared_dict["seaice_type"] == 3] = self.rho_myi

        thickness = (
            (shared_dict["snow_depth"] * shared_dict["snow_density"])
            + (shared_dict["freeboard_corr"] * ice_densities)
        ) / (ice_densities - self.rho_sea)

        self.log.info(
            "Thickness - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f Count=%d NaN=%d",
            np.nanmean(thickness),
            np.nanstd(thickness),
            np.nanmin(thickness),
            np.nanmax(thickness),
            thickness.shape[0],
            sum(np.isnan(thickness)),
        )

        shared_dict["thickness"] = thickness

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

        """ None """

        # ---------------------------------------------------------------------
