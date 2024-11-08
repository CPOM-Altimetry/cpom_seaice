"""clev2er.algorithms.seaice.alg_warren_snow_means.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds snow depth and density values for each record to shared_mem. These values are precomputed
and loaded from an auxilliary file 'warren_means.dat'.


#Main initialization (init() function) steps/resources required

Check warren_means.dat exists and is readable
Open file
read file data to memory
Close file

#Main process() function steps

Determine which records have an ice type of 2 (First year ice)
and which have a type of 3 (Multi-year ice)
Create snow_depth array of np.nans (all records with ice types other than 2 or 3 will remain np.nan)
Create snow_density array of np.nans
Determine the month of each measurement from the time
Set records with type 2 or 3 ice to the corresponding month's snow_depth mean
If ice_type is 2, divide snow_depth by 2
Set records with type 2 or 3 ice to the corresponding month's snow_density mean
Save snow_depth and snow_density to shared_mem

#Main finalize() function steps

None

#Contribution to shared_dict

snow_depth : np.ndarray[float] = Precomputed mean snow depth
snow_density : np.ndarray[float] = Precomputed mean snow density

#Requires from shared_dict

seaice_type

Author: Ben Palmer
Date: 04 Sep 2024
"""

import os
from datetime import datetime
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

        # Check warren_means.dat exists and is readable
        # Open file
        # read file data to memory
        # Close file

        warren_means_file_path = os.path.join(
            self.config["shared"]["aux_file_path"], "warren_means.dat"
        )

        self.log.info("\tLoading warren_means.dat...")
        if not os.path.exists(warren_means_file_path):
            raise FileNotFoundError(
                f"Cannot find warren_means.dat in {self.config['shared']['aux_file_path']}"
            )

        _, self.wm_depth, self.wm_density = np.transpose(np.genfromtxt(warren_means_file_path))

        self.log.info("\tLoaded data successfully!")

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

        # Determine which records have an ice type of 2 (First year ice)
        # and which have a type of 3 (Multi-year ice)
        # Create snow_depth array of np.nans (all records with ice types other than 2
        # or 3 will remain np.nan)
        # Create snow_density array of np.nans
        # Determine the month of each measurement from the time
        # Set records with type 2 or 3 ice to the corresponding month's snow_depth mean
        # If ice_type is 2, divide snow_depth by 2
        # Set records with type 2 or 3 ice to the corresponding month's snow_density mean
        # Save snow_depth and snow_density to shared_mem

        has_fyi = shared_dict["seaice_type"] == 2
        has_mfi = shared_dict["seaice_type"] == 3

        # set all values to nans, anything that isnt fyi or mfi will remain unset
        snow_depth = np.zeros(l1b["measurement_time"].shape[0]) * np.nan
        snow_density = np.zeros(l1b["measurement_time"].shape[0]) * np.nan

        # Need the month of each record to find the mean values
        # would use lambda but mypy and ruff do not like that, need to define function instead
        def ts_to_month(ts):
            return datetime.fromtimestamp(ts).month

        tsv = np.vectorize(ts_to_month)
        measurement_months = tsv(l1b["measurement_time"][:].data.astype(int)) - 1
        # Subtract 1 from the month so they match the indexes for the depth and density

        # get the depth values for fyi and myi
        snow_depth[(has_fyi | has_mfi)] = self.wm_depth[measurement_months][(has_fyi | has_mfi)]
        snow_depth[has_fyi] /= 2  # if fyi, divide depth by 2
        # get the density values for fyi and myi
        snow_density[(has_fyi | has_mfi)] = self.wm_density[measurement_months][(has_fyi | has_mfi)]

        # save to shared_dict
        shared_dict["snow_depth"] = snow_depth
        shared_dict["snow_density"] = snow_density

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

        # None

        # ---------------------------------------------------------------------
