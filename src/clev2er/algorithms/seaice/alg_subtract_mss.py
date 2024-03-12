""" clev2er.algorithms.seaice.alg_subtract_mss.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose

    Subtracts the mean sea surface from the elevations calculated in alg_elev_calculations.
    
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
from scipy.io import readsav
from scipy.spatial import cKDTree

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
        # pylint: disable=too-many-locals

        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        # Load MSS config
        mss_file_path = self.config["alg_subtract_mss"]["mss_file"]
        buffer = self.config["alg_subtract_mss"]["mss_buffer"]
        max_lat = self.config["alg_subtract_mss"]["max_latitude"]
        max_lon = self.config["alg_subtract_mss"]["max_longitude"]
        min_lat = self.config["alg_subtract_mss"]["min_latitude"]
        min_lon = self.config["alg_subtract_mss"]["min_longitude"]

        # Create projection transform
        crs_input = proj.Proj(self.config["alg_subtract_mss"]["input_projection"])
        crs_output = proj.Proj(self.config["alg_subtract_mss"]["output_projection"])
        self.lonlat_to_xy = proj.Transformer.from_proj(crs_input, crs_output, always_xy=True)

        # Load MSS file
        self.log.info("\tLoading MSS from %s", mss_file_path)
        if not os.path.exists(mss_file_path):
            self.log.error("Cannot find MSS file - %s", mss_file_path)
            raise RuntimeError(f"Cannot find the MSS file at {mss_file_path}")
        mss_all = readsav(mss_file_path)["mss"]

        # Filter MSS to correct area
        mss_filt = mss_all[
            (mss_all[:, 1] > min_lat - buffer)
            & (mss_all[:, 1] < max_lat + buffer)
            & (mss_all[:, 0] % 360 > min_lon - buffer)
            & (mss_all[:, 0] % 360 < max_lon + buffer)
        ]

        # Assemble KDTree
        mss_lat = mss_filt[:, 1]
        mss_lon = mss_filt[:, 0] % 360
        self.mss_vals = mss_filt[:, 2]

        mss_x, mss_y = self.lonlat_to_xy.transform(  # pylint: disable=unpacking-non-sequence
            mss_lon, mss_lat
        )
        mss_points = np.transpose((mss_x, mss_y))
        self.mss_tree = cKDTree(mss_points)

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

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        # transform lat lon values to points
        sample_x, sample_y = self.lonlat_to_xy.transform(  # pylint: disable=unpacking-non-sequence
            shared_dict["sat_lon"], shared_dict["sat_lat"]
        )
        sample_points = np.transpose((sample_x, sample_y))

        sample_mss_indices = np.apply_along_axis(self.mss_tree.query, 1, sample_points, k=1)[
            :, 1
        ].astype(int)

        self.log.info("Number of NaNs in MSS - %d", sum(np.isnan(sample_mss_indices)))

        shared_dict["mss"] = self.mss_vals[sample_mss_indices]

        sla = shared_dict["elevation"] - shared_dict["mss"]

        self.log.info("Number of NaNs in elevation_corrected - %d", sum(np.isnan(sla)))
        self.log.info(
            "SLA - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f",
            np.nanmean(sla),
            np.nanstd(sla),
            np.nanmin(sla),
            np.nanmax(sla),
        )

        shared_dict["sea_level_anomaly"] = sla

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
