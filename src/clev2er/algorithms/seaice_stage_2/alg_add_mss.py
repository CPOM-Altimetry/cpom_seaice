"""clev2er.algorithms.seaice_stage_1.alg_add_mss.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds the MSS values for each sample to the shared dict.

#Main initialization (init() function) steps/resources required

Read the MSS file location from config and load into a KDTree

#Main process() function steps

Match each sample to the correct MSS value
Subtract the MSS and retracker bias from the elevation for each sample

#Contribution to shared_dict

elevation_corrected (np.ndarray) : array of elevations after mss and retracker bias
    have been removed

#Requires from shared_dict

elevation
sat_lat
sat_lon

Author: Ben Palmer
Date: 06 Mar 2024
"""

import os
from typing import Tuple

import numpy as np
import pyproj as proj
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

    # pylint:disable=too-many-instance-attributes

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
        # pylint: disable=too-many-locals

        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        # Load MSS config
        mss_file_path = self.config["alg_add_mss"]["mss_file"]
        buffer = self.config["alg_add_mss"]["mss_buffer"]
        max_lat = self.config["shared"]["max_latitude"]
        max_lon = self.config["shared"]["max_longitude"]
        min_lat = self.config["shared"]["min_latitude"]
        min_lon = self.config["shared"]["min_longitude"]

        self.delta = self.config["alg_add_mss"]["delta"]
        self.lonmin = self.config["alg_add_mss"]["lonmin"]
        self.latmin = self.config["alg_add_mss"]["latmin"]
        self.nlats = self.config["alg_add_mss"]["nlats"]
        self.nlons = self.config["alg_add_mss"]["nlons"]

        # Create projection transform
        crs_input = proj.Proj(self.config["alg_add_mss"]["input_projection"])
        crs_output = proj.Proj(self.config["shared"]["output_projection"])
        self.lonlat_to_xy = proj.Transformer.from_proj(crs_input, crs_output, always_xy=True)

        # Load MSS file
        self.log.info("\tLoading MSS from %s", mss_file_path)
        if not os.path.exists(mss_file_path):
            self.log.error("Cannot find MSS file - %s", mss_file_path)
            raise RuntimeError(f"Cannot find the MSS file at {mss_file_path}")

        mss_file = np.transpose(np.genfromtxt(mss_file_path))

        mss_values = mss_file[2]
        mss_lat = mss_file[1]
        mss_lon = mss_file[0]
        mss_lon_adjusted = mss_lon % 360

        # Filter MSS to correct area
        inside_area = (
            (mss_lat > min_lat - buffer)
            & (mss_lat < max_lat + buffer)
            & (mss_lon_adjusted > min_lon - buffer)
            & (mss_lon_adjusted < max_lon + buffer)
        )

        # Assemble KDTree
        mss_lat = mss_lat[inside_area]
        mss_lon = mss_lon[inside_area]
        mss_vals = mss_values[inside_area]

        fdxlat = (((mss_lat - self.latmin) / self.delta) + 0.5).astype(int)
        fdxlon = (((mss_lon - self.lonmin) / self.delta) + 0.5).astype(int)

        if np.any((0 > fdxlat) & (fdxlat >= self.nlats)):
            self.log.error("fdxlat contains out of bounds values")
            raise RuntimeError("fdxlat out of bounds")
        if np.any((0 > fdxlon) & (fdxlon >= self.nlons)):
            self.log.error("fdxlon contains out of bounds values")
            raise RuntimeError("fdxlon out of bounds")

        self.mss_grid = np.zeros((self.nlats, self.nlons), dtype=np.float64)
        self.mss_grid[fdxlat, fdxlon] = mss_vals

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
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

        # pylint:disable=too-many-locals

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        sample_mss = np.zeros(l1b["sat_lat"][:].data.size) * np.nan

        for sample_i, (sample_lat, sample_lon) in enumerate(
            zip(l1b["sat_lat"][:].data, l1b["sat_lon"][:].data)
        ):
            # Get the fdx of lats and lons
            sample_fdxlat = (sample_lat - self.latmin) / self.delta
            sample_fdxlon = (sample_lon - self.lonmin) / self.delta

            # skip if we can't interpolate
            if (
                (sample_fdxlat < 0)
                or (sample_fdxlat >= self.nlats - 1)
                or (sample_fdxlon < 0)
                or (sample_fdxlon >= self.nlons - 1)
            ):
                continue

            # Do interpolation of mss in the area (lat is x, lon is y)
            # Get fraction of lats and lons
            frac_lats, _ = np.modf(sample_fdxlat)
            frac_lons, _ = np.modf(sample_fdxlon)

            # Convert to integers so we can use as indices
            sample_fdxlat = sample_fdxlat.astype(int)
            sample_fdxlon = sample_fdxlon.astype(int)

            # get mss values around indices
            mss_1 = self.mss_grid[sample_fdxlat, sample_fdxlon]
            mss_2 = self.mss_grid[sample_fdxlat + 1, sample_fdxlon]
            mss_3 = self.mss_grid[sample_fdxlat, sample_fdxlon + 1]
            mss_4 = self.mss_grid[sample_fdxlat + 1, sample_fdxlon + 1]

            sample_mss[sample_i] = (
                ((1 - frac_lats) * (1 - frac_lons) * mss_1)
                + (frac_lats * (1 - frac_lons) * mss_2)
                + ((1 - frac_lats) * frac_lons * mss_3)
                + (frac_lats * frac_lons * mss_4)
            )

        self.log.info(
            "MSS - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f Count=%d NaN=%d",
            np.nanmean(sample_mss),
            np.nanstd(sample_mss),
            np.nanmin(sample_mss),
            np.nanmax(sample_mss),
            sample_mss.shape[0],
            sum(np.isnan(sample_mss)),
        )

        shared_dict["mss"] = sample_mss

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
        # Add finalization steps here \/
        # ---------------------------------------------------------------------

        # None

        # ---------------------------------------------------------------------
