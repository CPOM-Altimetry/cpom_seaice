"""clev2er.algorithms.seaice.alg_add_region_mask.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds the a region mask to the shared_mem

#Main initialization (init() function) steps/resources required

Load params from config
Check if region_mask directory exists
Construct file path for region mask
Load region mask to memory

#Main process() function steps

Get mask values for every sample
Save to shared_mem

#Main finalize() function steps

Remove mask data from memory

#Contribution to shared_dict

region_mask : np.ndarray[float] = Region mask values for samples

#Requires from shared_dict

sat_lat
sat_lon

Author: Ben Palmer
Date: 06 Sep 2024
"""

import os
from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# pylint:disable=pointless-string-statement
# pylint:disable=too-many-locals


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

        """ Load params from config
        Construct file path for region mask
        Check if file path exists
        Load region mask to memory """

        region_mask_file_path = os.path.join(
            self.config["shared"]["aux_file_path"],
            "region_masks",
            f"region_mask_N{self.config['alg_add_region_mask']['mask_number']:02}.dat",
        )
        nlats = self.config["shared"]["grid_nlats"]
        nlons = self.config["shared"]["grid_nlons"]

        # Load region mask file
        self.log.info("\tLoading region mask from %s", region_mask_file_path)
        if not os.path.exists(region_mask_file_path):
            self.log.error("Cannot find region mask file - %s", region_mask_file_path)
            raise RuntimeError(f"Cannot find the region mask file at {region_mask_file_path}")
        region_mask_file = np.transpose(np.genfromtxt(region_mask_file_path))

        # read data

        file_lat_index = region_mask_file[0]
        file_lon_index = region_mask_file[1]
        file_values = region_mask_file[4]

        # Filter region mask to correct area
        in_area = (
            (file_lat_index >= 0)
            & (file_lat_index < nlats)
            & (file_lon_index >= 0)
            & (file_lon_index < nlons)
        )
        file_lat_index = file_lat_index[in_area]
        file_lon_index = file_lon_index[in_area]
        file_values = file_values[in_area]

        # Assemble region grid
        self.region_mask_grid = np.zeros((nlats, nlons)) * np.nan
        self.region_mask_grid[file_lat_index, file_lon_index] = file_values

        self.log.info(
            "Region Mask - Count=%d nTrue=%d",
            np.prod(self.region_mask_grid.shape),
            np.sum(self.region_mask_grid),
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

        """ Get grid and add to the shared memory"""

        shared_dict["region_mask"] = self.region_mask_grid

        # apply region mask to data
        shared_dict["thickness_grid"] *= shared_dict["region_mask"]
        shared_dict["volume_grid"] *= shared_dict["region_mask"]
        shared_dict["iceconc_grid"] *= shared_dict["region_mask"]
        shared_dict["area_grid"] *= shared_dict["region_mask"]

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

        """ Remove mask data from memory """

        # ---------------------------------------------------------------------
