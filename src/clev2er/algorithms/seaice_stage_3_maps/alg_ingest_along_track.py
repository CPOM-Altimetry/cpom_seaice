"""clev2er.algorithms.seaice.alg_ingest_along_track.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Ingest data from along track input file and add it to shared dictionary

#Main initialization (init() function) steps/resources required

Get variables to be ingested from config

#Main process() function steps

For each variable in config:
    Read variable data
    Apply filtering
    Add to shared dict

#Main finalize() function steps

None

#Contribution to shared_dict

Variable as named in config

#Requires from shared_dict

None

Author: Ben Palmer
Date: 23 Feb 2026
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
        # pylint:disable=pointless-string-statement
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ 
        Get config parameters
        Check that output directory exists
        """

        self.variables = self.config["alg_ingest_along_track"]["variables"]
        self.filtering_on = bool(self.config["alg_ingest_along_track"]["filtering_on"])

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(
        self,
        l1b: Dataset,
        shared_dict: dict,  # pylint:disable=unused-argument
    ) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint:disable=pointless-string-statement
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

        """ 
        For each variable in config:
            Read variable data
            Apply filtering if necessary
            Add to shared dict
        """

        for var_name in self.variables:
            if var_name not in l1b.variables:
                raise RuntimeError(f"Cannot find variable {var_name} in input file.")

            data = l1b[var_name][:].data

            # Variable specific filtering
            if self.filtering_on:
                match var_name:
                    case "thickness":
                        if "valid" not in l1b.variables:
                            raise RuntimeError(
                                "Input file must contain valid variable if filtering thickness"
                            )
                        sample_valid = l1b["valid"][:].data.flatten().astype(bool)
                        data[~sample_valid] = np.nan
                    case "freeboard":
                        outside_range = (data < -0.3) | (data > 3)
                        data[outside_range] = np.nan
                    case "seaice_conc":
                        outside_range = data < 15.0
                        data[outside_range] = np.nan
                    case "snow_depth":
                        if "freeboard" not in l1b.variables:
                            raise RuntimeError(
                                "Input file must contain freeboard variable if filtering snow_depth"
                            )
                        freeboard = l1b["freeboard"][:].data.flatten()
                        outside_range = (freeboard < -0.3) | (freeboard > 3)
                        data[outside_range] = np.nan
                    case _:
                        self.log.info("Variable %s does not have a unique filtering step.")

            shared_dict[var_name] = data

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
        # pylint:disable=pointless-string-statement

        # ---------------------------------------------------------------------
        # Add finalization steps here /
        # ---------------------------------------------------------------------

        """ None """

        # ---------------------------------------------------------------------
