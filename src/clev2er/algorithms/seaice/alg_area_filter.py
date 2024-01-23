""" clev2er.algorithms.seaice.alg_area_filter.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose
    
    Provides an area filter, using the latitude and longitude values
    in shared_dict['sat_lat'], shared_dict['sat_lon']

    #Main initialization (init() function) steps/resources required

    None

    #Main process() function steps

    

    #Contribution to shared_dict

    shared_dict["twice_ocean_tide_01"] (np.array[int]) : example contains 2 x the L1b ocean_tide_01 
                                                         parameter

    #Requires from shared_dict

    None
"""

from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm

# each algorithm shares some common class code, so pylint: disable=duplicate-code


class Algorithm(BaseAlgorithm):
    """CLEV2ER Algorithm class

    contains:
         .log (Logger) : log instance that must be used for all logging, set by BaseAlgorithm
         .config (dict) : configuration dictionary, set by BaseAlgorithm
         - functions that need completing:
         .init() : Algorithm initialization function (run once at start of chain)
         .process(l1b,shared_dict) : Algorithm processing function (run on every L1b file)
         .finalize() : Algorithm finalization/closure function (run after all chain
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

        self.min_latitude = self.config["alg_area_filter"]["min_latitude"]
        self.max_latitude = self.config["alg_area_filter"]["max_latitude"]
        self.min_longitude = self.config["alg_area_filter"]["min_longitude"]
        self.max_longitude = self.config["alg_area_filter"]["max_longitude"]

        # Test configuration settings
        if self.min_latitude < -90.0 or self.min_latitude > 90.0:
            raise ValueError(
                f"config[alg_area_filter][min_latitude] {self.min_latitude} out of range"
            )
        if self.max_latitude < -90.0 or self.max_latitude > 90.0:
            raise ValueError(
                f"config[alg_area_filter][max_latitude] {self.max_latitude} out of range"
            )
        if self.max_latitude < self.min_latitude:
            raise ValueError("config[alg_area_filter][max_latitude] should not be < [min_latitude]")
        if self.min_longitude < 0.0 or self.min_longitude > 360.0:
            raise ValueError(
                f"config[alg_area_filter][min_longitude] {self.min_longitude} out of range 0..360"
            )
        if self.max_longitude < 0.0 or self.max_longitude > 360.0:
            raise ValueError(
                f"config[alg_area_filter][max_longitude] {self.max_longitude} out of range 0..360"
            )
        if self.max_longitude < self.min_longitude:
            raise ValueError(
                "config[alg_area_filter][max_longitude] should not be < [min_longitude]"
            )

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

        # Creating boolean masks for each condition
        lat_condition = (shared_dict["sat_lat"] >= self.min_latitude) & (
            shared_dict["sat_lat"] <= self.max_latitude
        )
        lon_condition = (shared_dict["sat_lon"] >= self.min_longitude) & (
            shared_dict["sat_lon"] <= self.max_longitude
        )

        combined_condition = lat_condition & lon_condition

        # Getting the indices of the filtered values
        indices_inside = np.nonzero(lat_condition & lon_condition)[0]

        # Counting the number of points inside the area
        num_points_inside = len(indices_inside)
        total_points = len(shared_dict["sat_lon"])

        self.log.info("Number of points inside area = %d of %d", num_points_inside, total_points)
        self.log.info("%% of points inside area = %.2f%%", 100.0 * np.mean(combined_condition))

        if num_points_inside == 0:
            self.log.info("No points inside area filter")
            return (False, "SKIP_OK")  # Returning False with 'SKIP_OK' means no further algorithms
            # will be run for this L1b file, but that it is not an error
            # The chain will skip to the next L1b file (if there is one)

        # Outputs of the algorithm saved to the shared_dict

        shared_dict["num_points_inside_area"] = num_points_inside
        # filter the input parameter based on the area indices inside
        shared_dict["sat_lat"] = shared_dict["sat_lat"][indices_inside]
        shared_dict["sat_lon"] = shared_dict["sat_lon"][indices_inside]
        # Add other parameters to filter here

        # -------------------------------------------------------------------
        # Returns (True,'') if success
        return (success, error_str)

    def finalize(self, stage: int = 0) -> None:
        """Algorithm finalization function - called after all processing completed

          Can be used to clean up/free resources initialized in the init() function

        Args:
            stage (int, optional): this sets the stage when this function is called
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

        # No finalization required for this algorithm

        # ---------------------------------------------------------------------
