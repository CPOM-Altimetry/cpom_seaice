"""clev2er.algorithms.seaice.alg_dynamic_snow_load.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds snow depth and density values for each record to shared_mem. These values are calculated
from an external file by finding the mean of the 4 closest values to each sample and weighting the
mean by the distance to the centre of gravity


#Main initialization (init() function) steps/resources required

Read config options
Check that data folder exists

#Main process() function steps

For each sample
    Get date of sample
    Check if the correct file is loaded
        If not, load the new file
    Find the closest 4 cells
    Interpolate them weighted by their distance
Save snow_depth and snow_density to shared_mem

#Main finalize() function steps

None

#Contribution to shared_dict

snow_depth : np.ndarray[float] = Precomputed mean snow depth
snow_density : np.ndarray[float] = Precomputed mean snow density

#Requires from shared_dict

seaice_type

Author: Ben Palmer
Date: 17 Jun 2026
"""

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.time import Time
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import Proj, Transformer

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

    # pylint: disable=too-many-instance-attributes

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

        self.data_dir = Path(self.config["alg_dynamic_snow_load"]["data_dir"])
        self.grid_xymin = self.config["alg_dynamic_snow_load"]["grid_xymin"]
        self.grid_xymax = self.config["alg_dynamic_snow_load"]["grid_xymax"]
        self.grid_xystep = self.config["alg_dynamic_snow_load"]["grid_xystep"]
        self.grid_xynum = ((self.grid_xymax - self.grid_xymin) / self.grid_xystep) + 1
        self.cog_max = self.config["alg_dynamic_snow_load"]["cog_max"]

        self.log.info("\tLoading warren_means.dat...")
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Cannot find data directory {self.config['shared']['aux_file_path']}"
            )

        input_projection = self.config["alg_add_si_type"]["input_projection"]
        output_projection = self.config["shared"]["output_projection"]

        self.log.info(
            "Transforming projection from %s to %s for value reading",
            input_projection,
            output_projection,
        )

        crs_input = Proj(input_projection)
        crs_output = Proj(output_projection)
        self.lonlat_to_xy = Transformer.from_proj(crs_input, crs_output, always_xy=True)

        self.log.info("\tLoaded data successfully!")

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-instance-attributes
        # pylint: disable=too-many-statements
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

        snow_depth = np.full(l1b["sat_lat"][:].data.size, np.nan, dtype=np.float64)
        snow_load = np.full(l1b["sat_lat"][:].data.size, np.nan, dtype=np.float64)

        current_date = None

        sat_time = Time(l1b["measurement_time"][:].data, format="unix_tai")

        sat_x, sat_y = self.lonlat_to_xy.transform(l1b["sat_lon"][:].data, l1b["sat_lat"][:].data)

        for i, (sample_time, sample_x, sample_y) in enumerate(zip(sat_time, sat_x, sat_y)):
            file_date: date = sample_time.to_datetime().date()

            if current_date != file_date:
                # load new file
                self.log.info("Loading new file for %04d/%02d", file_date.year, file_date.month)

                file_path = (
                    self.data_dir
                    / f"{file_date.year:04d}"
                    / f"{file_date.year:04d}{file_date.month:02d}{file_date.day:02d}"
                )

                depth_path = file_path.with_suffix(".Depth")
                if not depth_path.exists():
                    self.log.error("Cannot find depth file at %s", depth_path)
                    raise FileNotFoundError(f"Cannot find depth file at {depth_path}")

                load_path = file_path.with_suffix(".Depth")
                if not load_path.exists():
                    self.log.error("Cannot find load file at %s", load_path)
                    raise FileNotFoundError(f"Cannot find load file at {load_path}")

                # load snow depth file
                file_depth = np.full((self.grid_xynum, self.grid_xynum), np.nan, dtype=np.float64)
                snow_depth_data = np.transpose(np.genfromtxt(str()))
                cog_in_bounds = snow_depth_data[7] < self.cog_max
                file_x = snow_depth_data[0][cog_in_bounds]
                file_y = snow_depth_data[1][cog_in_bounds]
                depth_data = snow_depth_data[4][cog_in_bounds]

                x_index = (file_x - self.grid_xymin) / self.grid_xystep
                y_index = (file_y - self.grid_xymin) / self.grid_xystep

                file_depth[x_index][y_index] = depth_data

                # load snow load file
                file_load = np.full((self.grid_xynum, self.grid_xynum), np.nan, dtype=np.float64)
                snow_load_data = np.transpose(np.genfromtxt(str(file_path.with_suffix(".Load"))))
                cog_in_bounds = snow_load_data[7] < self.cog_max
                file_x = snow_load_data[0][cog_in_bounds]
                file_y = snow_load_data[1][cog_in_bounds]
                load_data = snow_load_data[4][cog_in_bounds]

                x_index = (file_x - self.grid_xymin) / self.grid_xystep
                y_index = (file_y - self.grid_xymin) / self.grid_xystep

                file_load[x_index][y_index] = load_data

                current_date = file_date

            fx = (sample_x - self.grid_xymin) / self.grid_xystep
            fy = (sample_y - self.grid_xymin) / self.grid_xystep

            fx_r, fx_i = np.modf(fx)
            fy_r, fy_i = np.modf(fy)

            v1 = file_depth[fx_i][fy_i]
            v2 = file_depth[fx_i + 1][fy_i]
            v3 = file_depth[fx_i][fy_i + 1]
            v4 = file_depth[fx_i + 1][fy_i + 1]
            snow_depth[i] = (
                ((1 - fx_r) * (1 - fy_r) * v1)
                + (fx_r * (1 - fy_r) * v2)
                + ((1 - fx_r) * fy_r * v3)
                + (fx_r * fy_r * v4)
            ) / 1000

            v1 = file_load[fx_i][fy_i]
            v2 = file_load[fx_i + 1][fy_i]
            v3 = file_load[fx_i][fy_i + 1]
            v4 = file_load[fx_i + 1][fy_i + 1]
            snow_load[i] = (
                ((1 - fx_r) * (1 - fy_r) * v1)
                + (fx_r * (1 - fy_r) * v2)
                + ((1 - fx_r) * fy_r * v3)
                + (fx_r * fy_r * v4)
            ) / 1000

        snow_density = snow_load / snow_depth

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
