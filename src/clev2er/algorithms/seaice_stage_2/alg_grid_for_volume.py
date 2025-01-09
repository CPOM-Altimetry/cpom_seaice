"""clev2er.algorithms.seaice.alg_grid_for_volume.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Combines files into grids for each month

#Main initialization (init() function) steps/resources required

Read params from config

#Main process() function steps

Check if grid file for month exists
If not, create empty nc file and empty numpy arrays for thickness, conc, volume, thick_fyi,
thick_myi, fraction of fyi and myi, counts, and fill values
If it does, load existing values from grid file
Get grid index from location data
Add thickness and ice concentration values to relevant grid cells
Increment number of samples inside by 1 each time
Do the same with fyi and myi thickness
Save variables to output grid file

#Main finalize() function steps

None

#Contribution to shared_dict

grid_file_name

#Requires from shared_dict

requirements

Author: Ben Palmer
Date: 19 Sep 2024
"""

import os
from datetime import datetime
from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# pylint:disable=pointless-string-statement


class Algorithm(BaseAlgorithm):
    # pylint:disable=too-many-instance-attributes
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

        """ Read params from config """
        self.nlats = self.config["shared"]["grid_nlats"]
        self.nlons = self.config["shared"]["grid_nlons"]
        self.grid_directory = self.config["alg_grid_for_volume"]["grid_directory"]

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
        # pylint:disable=too-many-statements
        # pylint:disable=too-many-branches

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # /    down the chain in the 'shared_dict' dict     /
        # -------------------------------------------------------------------

        """ 
            Check if grid file for month exists
            If not, create empty nc file and empty numpy arrays for thickness, conc, volume, 
            thick_fyi, thick_myi, fraction of fyi and myi, counts, and fill values
            If it does, load existing values from grid file
            Get grid index from location data
            Add thickness and ice concentration values to relevant grid cells
            Increment number of samples inside by 1 each time
            Do the same with fyi and myi thickness
            Save variables to output grid file
        """

        f_time = datetime.fromtimestamp(np.min(l1b["measurement_time"])).strftime("%Y%M")
        grid_file_name = f"{f_time}_grids.nc"
        grid_file_path = os.path.join(self.grid_directory, grid_file_name)

        # check if grids are already saved
        if os.path.exists(grid_file_path):
            # load grids from npz
            output_nc: Dataset = Dataset(grid_file_path, mode="a")
            thickness = output_nc["thickness"][:].data
            thickness_fyi = output_nc["thickness_fyi"][:].data
            thickness_myi = output_nc["thickness_myi"][:].data
            iceconc = output_nc["iceconc"][:].data
            nin = output_nc["number_in"][:].data
            nin_fyi = output_nc["number_in_fyi"][:].data
            nin_myi = output_nc["number_in_myi"][:].data
        else:
            # initialise all the arrays we need
            output_nc: Dataset = Dataset(grid_file_path, mode="w")  # type: ignore
            output_nc.createDimension("lat", self.nlats)
            output_nc.createDimension("lon", self.nlons)

            output_nc.createVariable("thickness", "f8", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("thickness_fyi", "f8", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("thickness_myi", "f8", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("iceconc", "f8", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("number_in", "i4", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("number_in_fyi", "i4", ("lat", "lon"), compression="zlib")
            output_nc.createVariable("number_in_myi", "i4", ("lat", "lon"), compression="zlib")

            output_nc.fdate = f_time

            thickness = np.zeros((self.nlats, self.nlons), dtype=np.float64)
            thickness_fyi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
            thickness_myi = np.zeros((self.nlats, self.nlons), dtype=np.float64)
            iceconc = np.zeros((self.nlats, self.nlons), dtype=np.float64)
            nin = np.zeros((self.nlats, self.nlons), dtype=np.int64)
            nin_fyi = np.zeros((self.nlats, self.nlons), dtype=np.int64)
            nin_myi = np.zeros((self.nlats, self.nlons), dtype=np.int64)

        # calculate the indexes of each record
        ilats = ((l1b["sat_lat"][:].data - 40) / 0.1).astype(int)
        ilons = (((l1b["sat_lon"][:].data + 180) % 360) / 0.5).astype(int)
        # line above is weird. Andy uses coordinates -180..180, we use 0..360
        # but andy's ilon formula only works right now with -180..180
        # to convert from former to latter, use: (lon + 180) % 360 - 180
        # andy's formula for lon to ilon is: (lon + 180) / 0.5
        # substituting the coordinate conversion into the ilon conversion, we get
        # ilon = ((lon + 180) % 360 - 180 + 180) / 0.5
        # can simplify as
        # ilon = ((lon + 180) % 360) / 0.5
        # you can probably simplify this by changing the ilon formula to work with
        # 0..360 values, but thats a problem for another day

        # add thickness data to array with indexes
        thickness[ilats, ilons] += shared_dict["thickness"]
        iceconc[ilats, ilons] += l1b["seaice_conc"][:].data
        nin[ilats, ilons] += 1

        sample_fyi = shared_dict["seaice_type"] == 2
        sample_myi = shared_dict["seaice_type"] == 3

        # calculate values for mfy and fyi
        thickness_fyi[ilats, ilons][sample_fyi] += shared_dict["thickness"][sample_fyi]
        nin_fyi[ilats, ilons][sample_fyi] += 1

        thickness_myi[ilats, ilons][sample_myi] += shared_dict["thickness"][sample_myi]
        nin_myi[ilats, ilons][sample_myi] += 1

        # save grids to nc file
        output_nc["iceconc"] = iceconc
        output_nc["thickness"] = thickness
        output_nc["thick_fyi"] = thickness_fyi
        output_nc["thick_myi"] = thickness_myi
        output_nc["nin_fyi"] = nin_fyi
        output_nc["nin_myi"] = nin_myi
        output_nc["nin"] = nin

        output_nc.close()

        self.log.info("Added data to grid %s", grid_file_name)

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

        """ finalize """

        # ---------------------------------------------------------------------
