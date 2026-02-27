"""clev2er.algorithms.seaice.alg_output_maps.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Outputs the gridded results of the processing chain in ASCII .map and .map.complete file formats,
the same format as the original chain.
This algorithm will dynamically save variables to ASCII files depending on what is 
specified in the config files.

#Main initialization (init() function) steps/resources required

Get config parameters
Check that output directory exists

#Main process() function steps

Get year and month of input file
If it does not exist, create folder for year and for outputs
Open output file as a text file
Loop through lats and lons
If cell isn't valid, don't print anything for .map file
Print line for .map.complete file

#Main finalize() function steps

None

#Contribution to shared_dict

None

#Requires from shared_dict

grid_x 
grid_y 
grid_lat 
grid_lon 
grid_mask 

For each variable:
    ..._grid
    ..._std
    ..._n_points
    ..._distance_from_cog

Author: Ben Palmer
Date: 23 Feb 2026
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.time import Time
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

        self.output_directory = Path(self.config["alg_output_maps"]["output_directory"])

        self.variables = self.config["alg_output_maps"]["variables"]

        if not self.output_directory.exists():
            os.makedirs(self.output_directory)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
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
        Get year and month of input file
        If it does not exist, create folder for year and for outputs
        Open output file as text files
        Loop through lats and lons
        Write line containing relevant information
        """

        f_time = Time(np.min(l1b["measurement_time"]), format="unix_tai").strftime("%Y%m")

        # check if year folder exists
        year_folder = self.output_directory / f_time[:4]
        if not year_folder.exists():
            os.makedirs(year_folder)

        filename_date = f"{f_time[:4]}_{f_time[4:]}"

        grid_x = shared_dict["grid_x"].flatten()
        grid_y = shared_dict["grid_y"].flatten()
        grid_lat = shared_dict["grid_lat"].flatten()
        grid_lon = shared_dict["grid_lon"].flatten()
        grid_mask = shared_dict["grid_mask"].flatten()

        for var_name in self.variables:
            output_varname = "".join([x.capitalize() for x in var_name.split("_")])
            self.log.info("Writing output for %s", var_name)
            map_file = os.path.join(year_folder, filename_date + "." + output_varname + ".map")
            map_complete_file = map_file + ".complete"

            var_values = shared_dict[var_name + "_grid"].flatten()
            var_std = shared_dict[var_name + "_std"].flatten()
            var_npoints = shared_dict[var_name + "_n_points"].flatten()
            var_cog_distance = shared_dict[var_name + "_distance_from_cog"].flatten()

            with (
                open(map_file, mode="+w", encoding="ascii") as map_fp,
                open(map_complete_file, mode="+w", encoding="ascii") as map_complete_fp,
            ):
                for i in range(len(shared_dict["grid_x"].flatten())):
                    x = int(grid_x[i])
                    y = int(grid_y[i])
                    n_points = int(var_npoints[i])

                    map_complete_fp.write(
                        f"{x: >9d}{y: >9d}"
                        f"{grid_lat[i]: >12.6f}{grid_lon[i]: >12.6f}"
                        f"{var_values[i]: >10.4f}"
                        f"{var_std[i]: >10.4f}"
                        f"{n_points: >10d}"
                        f"{var_cog_distance[i]: >15.4f}"
                        "\n"
                    )
                    # "%5d%5d%12.6f%12.6f%10.4f%10.4f%10d%10.4f\n"

                    if not grid_mask[i]:
                        map_fp.write(
                            f"{grid_lat[i]: >12.6f}{grid_lon[i]: >12.6f}"
                            f"{var_values[i]: >10.4f}"
                            f"{var_std[i]: >10.4f}"
                            f"{n_points: >10d}"
                            f"{var_cog_distance[i]: >15.4f}"
                            "\n"
                        )
                    # "%12.6f%12.6f%10.4f%10.4f%10d%10.4f\n"

            self.log.info("Saved map to %s", map_file)
            self.log.info("Saved complete map to %s", map_complete_file)

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
