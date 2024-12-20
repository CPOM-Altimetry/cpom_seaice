"""clev2er.algorithms.seaice.alg_add_ocean_frac.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds the ocean fraction data from auxilliary file to shared_mem

#Main initialization (init() function) steps/resources required

Read params from config
Check ocean fraction file exists
Read data from file
Prepare KDTree from data and save to algorithm memory

#Main process() function steps

For each sample, get the closest matching ocean fraction value

#Main finalize() function steps

None

#Contribution to shared_dict

ocean_frac : np.ndarray(float) = Array of ocean fraction values

#Requires from shared_dict

sat_lat
sat_lon
measurement_time

Author: Ben Palmer
Date: 09 Sep 2024
"""

import os
from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# pylint:disable=pointless-string-statement


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
        # pylint:disable=unpacking-non-sequence
        """ Read params from config
        Check ocean fraction file exists
        Read data from file
        Prepare KDTree from data and save to algorithm memory """

        # Load params from config
        ocean_frac_file_path = os.path.join(
            self.config["shared"]["aux_file_path"], "ocean_fraction_file_N.dat"
        )
        nlats = self.config["shared"]["grid_nlats"]
        nlons = self.config["shared"]["grid_nlons"]

        # Load ocean fraction file
        self.log.info("\tLoading ocean fraction from %s", ocean_frac_file_path)
        if not os.path.exists(ocean_frac_file_path):
            self.log.error("Cannot find ocean fraction file - %s", ocean_frac_file_path)
            raise RuntimeError(f"Cannot find the ocean fraction file at {ocean_frac_file_path}")

        # read file data
        # Wisdom from Andy Ridout
        # Input and output files have the following format:
        #     Col 1 : Latitude index
        #     Col 2 : Longitude index
        #     Col 3 : Latitude  of cell centre
        #     Col 4 : Longitude of cell centre
        #     Col 5 : Stored quantity

        ocean_frac_file = np.transpose(np.genfromtxt(ocean_frac_file_path))
        ocean_frac_lat = ocean_frac_file[0]
        ocean_frac_lon = ocean_frac_file[1]
        ocean_frac_values = ocean_frac_file[2]

        ocean_frac_lat_index = ((ocean_frac_lat - 40) / 0.1).astype(int)
        ocean_frac_lon_index = ((ocean_frac_lon + 180) / 0.5).astype(int)

        # remove values outside of the target area
        values_in_area = (
            (ocean_frac_lat_index >= 0)
            & (ocean_frac_lat_index < nlats)
            & (ocean_frac_lon_index >= 0)
            & (ocean_frac_lon_index < nlons)
        )
        ocean_frac_lat_index = ocean_frac_lat_index[values_in_area]
        ocean_frac_lon_index = ocean_frac_lon_index[values_in_area]
        ocean_frac_values = ocean_frac_values[values_in_area]

        # construct the grid
        self.ocean_frac_grid = np.zeros((nlats, nlons)) * np.nan
        self.ocean_frac_grid[ocean_frac_lat_index, ocean_frac_lon_index] = ocean_frac_values

        self.log.info(
            "Ocean Fraction - shape=%s Min=%f Mean=%f Max=%f NaNs=%d",
            self.ocean_frac_grid.shape,
            np.nanmin(self.ocean_frac_grid),
            np.nanmean(self.ocean_frac_grid),
            np.nanmax(self.ocean_frac_grid),
            np.sum(np.isnan(self.ocean_frac_grid.flatten())),
        )

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

        # Just get the grid and add it to the shared memory
        shared_dict["ocean_frac"] = self.ocean_frac_grid

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
