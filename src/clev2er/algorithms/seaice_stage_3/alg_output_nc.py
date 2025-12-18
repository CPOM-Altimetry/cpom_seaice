"""clev2er.algorithms.seaice.alg_output_nc.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Outputs the gridded results of the processing chain in NETCDF format.

#Main initialization (init() function) steps/resources required

Get config parameters
Check that output directory exists

#Main process() function steps

Get year and month of input file
If it does not exist, create folder for year and for outputs
Open output file as a netcdf4 Dataset
Add all needed variables

#Main finalize() function steps

None

#Contribution to shared_dict

None

#Requires from shared_dict

shared_dict["frac_fyi_grid"]
shared_dict["frac_myi_grid"]
shared_dict["iceconc_grid"]
shared_dict["number_in"]
shared_dict["thickness_grid"]
shared_dict["volume_grid"]

Author: Ben Palmer
Date: 18 Dec 2025
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    # pylint: disable=too-many-instance-attributes
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

        self.nlats = self.config["shared"]["nlats"]
        self.nlons = self.config["shared"]["nlons"]
        self.output_directory = Path(self.config["alg_output_nc"]["output_directory"])
        self.minimum_latitude = self.config["shared"]["min_latitude"]
        self.maximum_latitude = self.config["shared"]["max_latitude"]
        self.minimum_longitude = self.config["shared"]["min_longitude"]
        self.maximum_longitude = self.config["shared"]["max_longitude"]

        if not self.output_directory.exists():
            os.makedirs(self.output_directory)

        # make lat and lon grids
        lats = 40 + np.arange(0, self.nlats) * 0.1
        lons = -180 + np.arange(0, self.nlons) * 0.5

        self.lon_grid, self.lat_grid = np.meshgrid(lons, lats)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
        # pylint: disable=too-many-instance-attributes
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
        Get year and month of input file
        If it does not exist, create folder for year and for outputs
        Open output file as text files
        Loop through lats and lons
        Write line containing relevant information
        """

        # check if year folder exists
        year_folder = self.output_directory / l1b.f_time[:4]
        if not year_folder.exists():
            os.makedirs(year_folder)

        filename = f"CPOM_SI_CS2_{l1b.f_time}.nc"

        with Dataset(filename, mode="w") as out_nc:
            # attributes
            out_nc.year = int(l1b.f_time[:4])
            out_nc.month = int(l1b.f_time[4:])
            out_nc.source_mission = "CryoSat-2"

            # VARIABLES
            # dimensions
            out_nc.createDimension("lats", self.nlats)
            out_nc.createDimension("lons", self.nlons)

            # latitude
            lat_var = out_nc.createVariable("latitude", "f8", ("lats", "lons"))
            lat_var.units = "degrees north"
            lat_var.min_value = self.minimum_latitude
            lat_var.max_value = self.maximum_latitude
            lat_var[:] = self.lat_grid

            # longitude
            lon_var = out_nc.createVariable("longitude", "f8", ("lats", "lons"))
            lon_var.units = "degrees east"
            lon_var.min_value = self.minimum_longitude
            lon_var.max_value = self.maximum_longitude
            lon_var[:] = self.lon_grid

            # thickness
            thickness_var = out_nc.createVariable("thickness", "f8", ("lats", "lons"))
            thickness_var.units = "m"
            thickness_var[:] = shared_dict["thickness_grid"]

            # volume
            volume_var = out_nc.createVariable("volume", "f8", ("lats", "lons"))
            volume_var.units = "m^3"
            volume_var[:] = shared_dict["volume_grid"]

            # number of samples in gridcell
            nin_var = out_nc.createVariable("n_samples", "i4", ("lats", "lons"))
            nin_var[:] = shared_dict["number_in"]

            # concentration
            conc_var = out_nc.createVariable("ice_concentration", "f8", ("lats", "lons"))
            conc_var[:] = shared_dict["iceconc_grid"]

            # fraction of fyi thickness
            frac_fyi_var = out_nc.createVariable("fraction_fyi", "f8", ("lats", "lons"))
            frac_fyi_var[:] = shared_dict["frac_fyi_grid"]

            # fraction of myi thickness
            frac_fyi_myi = out_nc.createVariable("fraction_myi", "f8", ("lats", "lons"))
            frac_fyi_myi[:] = shared_dict["frac_myi_grid"]

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
