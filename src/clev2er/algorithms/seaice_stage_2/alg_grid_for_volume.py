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
from clev2er.utils.gridding.gridding import GriddedDataFile, VariableSpec

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

        # Define variables for the gridded data file
        self.variable_specs = [
            VariableSpec("thickness", "f8", ("lat", "lon"), compression="zlib"),
            VariableSpec("thickness_fyi", "f8", ("lat", "lon"), compression="zlib"),
            VariableSpec("thickness_myi", "f8", ("lat", "lon"), compression="zlib"),
            VariableSpec("iceconc", "f8", ("lat", "lon"), compression="zlib"),
            VariableSpec("number_in", "i4", ("lat", "lon"), compression="zlib"),
            VariableSpec("number_in_fyi", "i4", ("lat", "lon"), compression="zlib"),
            VariableSpec("number_in_myi", "i4", ("lat", "lon"), compression="zlib"),
        ]

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

        # Set up output file
        f_time = datetime.fromtimestamp(np.min(l1b["measurement_time"]).astype(int)).strftime(
            "%Y%m"
        )
        grid_file_name = f"{f_time}_grids.nc"
        grid_file_path = os.path.join(self.grid_directory, grid_file_name)

        with GriddedDataFile(
            variables=self.variable_specs,
            filename=grid_file_path,
            nrows=self.nlons,
            ncols=self.nlats,
        ) as gdf:
            if "f_time" not in gdf.get_attributes():
                gdf.add_attributes({"f_time", f_time})

            sample_fyi = shared_dict["seaice_type"] == 2
            sample_myi = shared_dict["seaice_type"] == 3

            gdf.grid_points(
                coordinates={"lat": l1b["sat_lat"][:].data, "lon": l1b["sat_lon"][:].data},
                data={
                    "thickness": shared_dict["thickness"],
                    "iceconc": l1b["seaice_conc"][:].data,
                    "number_in": 1,
                    "thickness_fyi": shared_dict["thickness"],
                    "number_in_fyi": 1,
                    "thickness_myi": shared_dict["thickness"],
                    "number_in_myi": 1,
                },
                conditions={
                    "thickness_fyi": sample_fyi,
                    "number_in_fyi": sample_fyi,
                    "thickness_myi": sample_myi,
                    "number_in_myi": sample_myi,
                },
            )

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
