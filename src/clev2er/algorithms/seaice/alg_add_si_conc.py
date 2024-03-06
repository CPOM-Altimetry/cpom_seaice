""" clev2er.algorithms.seaice.alg_add_si_conc.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose

    Gets the seaice concentration data from an external file and adds the required values 
    for the samples being processed. To prevent repeatedly loading in the same file every time
    .process() is called, keep a dict for the most recent file to store the KDTree for lat,lon 
    pairs and concentration values. Before the file is loaded in, it checks to see if the filename 
    is within the dict. If it is, use those values. If not, load the file and add the filename and 
    values to the dict.
    
    The KDTrees are stored instead of latitude and longitude values to prevent repeat processing 
    of creating the KDTree when values are the same, since creating the KDTree takes as much time as
    reading the file if not longer. 

    #Main initialization (init() function) steps/resources required

    Create an algorithm memory for loading files.
    Set config for seaice concentration file directory
    Set config for input and output projections 
    Create projection transformer

    #Main process() function steps

    Use the date of the timestamp of each sample to find which file to use.
    Load in the file / read from the memory dict
    convert lat lon to x y points 
    convert poitns to KDTree
    match points in sample to nearest point in KDTree
    find the value that corresponds to the nearest point
    save list of values to shared_dict
    
    #Main finalize() function steps
    Clear most recent file memory
    Delete latlon to xy transformer

    #Contribution to shared_dict

    'seaice_concentrations' (np.NDArray[float]) : Array of seaice concentration values for each 
        sample

    #Requires from shared_dict

    'sat_lat'
    'sat_lon'
    'measurement_time'

    Author: Ben Palmer
    Date: 01 Mar 2024
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

        # Store the data for the most recent file with this
        self.most_recent_file: Dict = {"date": ""}

        self.conc_file_dir = self.config["alg_add_si_conc"]["conc_file_dir"]

        input_projection = self.config["alg_add_si_conc"]["input_projection"]
        output_projection = self.config["alg_add_si_conc"]["output_projection"]

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
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        si_concentration = np.zeros(shared_dict["sat_lat"].size) * np.nan

        # for each timestamp, lat and lon in shared memory:
        for wv_num, (wv_timestamp, wv_lat, wv_lon) in enumerate(
            zip(shared_dict["measurement_time"], shared_dict["sat_lat"], shared_dict["sat_lon"])
        ):
            file_date = datetime.fromtimestamp(wv_timestamp).strftime("%Y%m%d")

            if self.most_recent_file["date"] == file_date:
                # If date is the same as the most recent file date, get values from dict
                file_point_tree = self.most_recent_file["tree"]
                file_values = self.most_recent_file["values"]

            else:
                # Else, read the file, create the KDTree and store the values
                # in most recent file dict for later use
                self.log.info("Loading new concentration data file  - %s", file_date)

                # Find the correct file for the data

                file_paths = glob.glob(os.path.join(self.conc_file_dir, f"nt_{file_date}*.dat"))

                # There should be 1 match for each date. If not, return an error
                if len(file_paths) < 1:
                    self.log.error("Could not locate file matching - %s", file_date)
                    return (False, "CONC_FILE_NOT_FOUND")
                if len(file_paths) > 1:
                    self.log.error(
                        "Too many files found that match - %s. Only one should be found.", file_date
                    )
                    return (False, "CONC_FILE_TOO_MANY_FOUND")

                file_path = file_paths[0]

                # Read the external file

                sea_ice_conc = np.transpose(np.genfromtxt(file_path))
                file_lats = sea_ice_conc[2]
                # convert to 0..360 to match shared_dict values
                file_lons = sea_ice_conc[3] % 360.0
                file_values = sea_ice_conc[4]

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

            wv_x, wv_y = self.lonlat_to_xy.transform(wv_lon, wv_lat)

            file_neighbouring_indices = int(file_point_tree.query((wv_x, wv_y), k=1)[1])
            si_concentration[wv_num] = file_values[file_neighbouring_indices]

        self.log.info("NaNs in concentration array - %d", sum(np.isnan(si_concentration)))

        shared_dict["seaice_concentration"] = si_concentration

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

        # clear file memory and remove lonlat transformer
        self.most_recent_file.clear()
        del self.lonlat_to_xy

        # ---------------------------------------------------------------------
