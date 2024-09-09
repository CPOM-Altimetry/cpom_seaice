"""clev2er.algorithms.seaice.alg_add_cell_area.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds the cell area data from auxilliary file to shared_mem

#Main initialization (init() function) steps/resources required

Read params from config
Check cell area file exists
Read data from file
Prepare KDTree from data and save to algorithm memory

#Main process() function steps

For each sample, get the closest matching cell area value

#Main finalize() function steps

None

#Contribution to shared_dict

cell_area : np.ndarray(float) = Array of cell area values

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
import pyproj as proj
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from scipy.spatial import cKDTree

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
        Check cell area file exists
        Read data from file
        Prepare KDTree from data and save to algorithm memory """

        # Load params from config
        cell_area_file_path = os.path.join(
            self.config["shared"]["aux_file_path"], "cell_area_file.dat"
        )
        max_lat = self.config["shared"]["max_latitude"]
        max_lon = self.config["shared"]["max_longitude"]
        min_lat = self.config["shared"]["min_latitude"]
        min_lon = self.config["shared"]["min_longitude"]

        # Create projection transform
        crs_input = proj.Proj(self.config["alg_add_cell_area"]["input_projection"])
        crs_output = proj.Proj(self.config["shared"]["output_projection"])
        self.lonlat_to_xy = proj.Transformer.from_proj(crs_input, crs_output, always_xy=True)

        # Load cell area file
        self.log.info("\tLoading cell area from %s", cell_area_file_path)
        if not os.path.exists(cell_area_file_path):
            self.log.error("Cannot find cell area file - %s", cell_area_file_path)
            raise RuntimeError(f"Cannot find the cell area file at {cell_area_file_path}")
        cell_area_file = np.transpose(np.genfromtxt(cell_area_file_path))
        cell_area_lats = cell_area_file[0]
        # convert to 0..360 to match shared_dict values
        cell_area_lons = cell_area_file[1] % 360.0

        # remove values outside of the target area
        values_in_area = (
            (cell_area_lats > min_lat)
            & (cell_area_lats < max_lat)
            & (cell_area_lons > min_lon)
            & (cell_area_lons < max_lon)
        )
        cell_area_lats = cell_area_lats[values_in_area]
        cell_area_lons = cell_area_lons[values_in_area]
        self.cell_area_values = cell_area_file[2][values_in_area]

        # construct the KDTree
        cell_area_x, cell_area_y = self.lonlat_to_xy.transform(cell_area_lons, cell_area_lats)
        cell_area_points = np.transpose((cell_area_x, cell_area_y))
        self.cell_area_tree = cKDTree(cell_area_points)

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

        """ For each sample, get the closest matching cell area value """
        sample_x, sample_y = self.lonlat_to_xy.transform(  # pylint: disable=unpacking-non-sequence
            l1b["sat_lon"][:].data, l1b["sat_lat"][:].data
        )
        sample_points = np.transpose((sample_x, sample_y))

        sample_cell_area_indices = np.apply_along_axis(
            self.cell_area_tree.query, 1, sample_points, k=1
        )[:, 1].astype(int)

        shared_dict["cell_area"] = self.cell_area_values[sample_cell_area_indices].astype(float)
        self.log.info(
            "Cell Area - Count=%d Min=%f Mean=%f Max=%f",
            shared_dict["cell_area"].shape[0],
            np.min(shared_dict["cell_area"]),
            np.mean(shared_dict["cell_area"]),
            np.max(shared_dict["cell_area"]),
        )

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
