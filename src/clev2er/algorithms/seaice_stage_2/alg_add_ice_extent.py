"""clev2er.algorithms.seaice.alg_add_ice_extent.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Adds an ice extent mask to the shared_mem

#Main initialization (init() function) steps/resources required

Load params from config
Check auxilliary file directory exists
Create place to keep auxilliary file data in memory between files/records

#Main process() function steps

Create array for extent mask
For each record:
    Check if loaded data is relevant
    If not, load relevant data to memory
    Find the concentration for that lat/lon pair
    If greater than threshold, value of mask is true, else it is false
Save the mask to shared_mem

#Main finalize() function steps

None

#Contribution to shared_dict

extent_mask

#Requires from shared_dict

measurement_time
sat_lat
sat_lon

Author: Ben Palmer
Date: 05 Sep 2024
"""

import glob
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pyproj as proj
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import Transformer
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

        """ Load params from config
        Check auxilliary file directory exists
        Create place to keep auxilliary file data in memory between files/records
        """

        self.conc_threshold = self.config["alg_add_ice_extent"]["conc_threshold"]

        # Store the data for the most recent file with this
        self.most_recent_file: Dict = {"date": ""}

        self.extent_file_dir = os.path.join(
            self.config["shared"]["aux_file_path"], "ice_extent_north"
        )

        input_projection = self.config["alg_add_ice_extent"]["input_projection"]
        output_projection = self.config["shared"]["output_projection"]

        self.log.info(
            "Transforming projection from %s to %s for value reading",
            input_projection,
            output_projection,
        )

        crs_input = proj.Proj(input_projection)
        crs_output = proj.Proj(output_projection)
        self.lonlat_to_xy = Transformer.from_proj(crs_input, crs_output, always_xy=True)

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

        """Create array for extent mask
        For each record:
            Check if loaded data is relevant
            If not, load relevant data to memory
            Find the concentration for that lat/lon pair
            If greater than threshold, value of mask is true, else it is false
        Save the mask to shared_mem
        """

        si_extent = np.zeros(l1b["sat_lat"][:].size, dtype=bool)

        # for each timestamp, lat and lon in shared memory:
        for sample_num, (sample_timestamp, sample_lat, sample_lon) in enumerate(
            zip(l1b["measurement_time"][:], l1b["sat_lat"][:], l1b["sat_lon"][:])
        ):
            file_date = datetime.fromtimestamp(int(sample_timestamp)).strftime("%Y%m%d")

            if self.most_recent_file["date"] == file_date:
                # If date is the same as the most recent file date, get values from dict
                file_point_tree = self.most_recent_file["tree"]
                file_values = self.most_recent_file["values"]

            else:
                # Else, read the file, create the KDTree and store the values
                # in most recent file dict for later use
                self.log.info("Loading new extent data file  - %s", file_date)

                # Find the correct file for the data

                file_paths = glob.glob(os.path.join(self.extent_file_dir, f"*{file_date}*.dat"))

                # There should be 1 match for each date. If not, return an error
                if len(file_paths) < 1:
                    self.log.error("Could not locate file matching - %s", file_date)
                    return (False, "EXTENT_FILE_NOT_FOUND")
                if len(file_paths) > 1:
                    self.log.error(
                        "Too many files found that match - %s. Only one should be found.", file_date
                    )
                    return (False, "EXTENT_FILE_TOO_MANY_FOUND")

                file_path = file_paths[0]

                # Read the external file

                sea_ice_extent = np.transpose(np.genfromtxt(file_path))
                file_lats = sea_ice_extent[2]
                # convert to 0..360 to match shared_dict values
                file_lons = sea_ice_extent[3] % 360.0
                file_values = sea_ice_extent[4]

                # Convert the longitudes and latitudes to (x, y) pairs and create a KDTree of points
                file_x, file_y = self.lonlat_to_xy.transform(file_lons, file_lats)
                file_points = np.transpose((file_x, file_y))
                file_point_tree = cKDTree(file_points)

                # Save the loaded date, KDTree, and values
                # Faster to save the tree than save the lon + lat values and recreate
                # the tree every time

                self.most_recent_file["date"] = file_date
                self.most_recent_file["tree"] = file_point_tree
                self.most_recent_file["values"] = file_values

            sample_x, sample_y = self.lonlat_to_xy.transform(sample_lon, sample_lat)

            file_neighbouring_indices = int(file_point_tree.query((sample_x, sample_y), k=1)[1])
            si_extent[sample_num] = file_values[file_neighbouring_indices] >= self.conc_threshold

        self.log.info("Ice Extent Mask - Count=%d nTrue=%d", si_extent.shape[0], np.sum(si_extent))

        shared_dict["extent_mask"] = si_extent

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
