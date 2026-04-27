"""clev2er.algorithms.seaice.floe_chord_length.py


Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Grids points using latitude and longitude by finding all points within a set radius of each
grid cell. Uses the mean of found points.
Uses a grid defined by an external file.

#Main initialization (init() function) steps/resources required

Get config parameters

#Main process() function steps

make floe chord length array
for each index in alongtrack data:
    skip if not valid
    if first valid thickness sample:
        save index

    if found first valid thickness:
        check distance vs first index lat/lon
        if under max distance:
            keep running total of thicknesses
            keep running number of thicknesses
        if larger than max distance:
            calculate mean thickness from running totals
            set first index to distance
            reset running totals, including current values



#Main finalize() function steps

None

#Contribution to shared_dict

floe_chord_length

#Requires from shared_dict

None

Author: Ben Palmer
Date: 20 Feb 2026
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from sklearn.metrics.pairwise import haversine_distances

from clev2er.algorithms.base.base_alg import BaseAlgorithm


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
        # pylint:disable=pointless-string-statement
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ 
        Get config parameters
        """

        self.max_distance = self.config["alg_floe_chord_length"]["max_distance"]
        self.earth_radius = self.config["geophysical"]["earth_radius"]

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
        # pylint: disable=too-many-statements
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
        make floe chord length array
        for each index in alongtrack data:
            skip if not valid
            if first valid thickness sample:
                save index
            
            if found first valid thickness:
                check distance vs first index lat/lon
                if under max distance:
                    keep running total of thicknesses
                    keep running number of thicknesses
                if larger than max distance:
                    calculate mean thickness from running totals
                    set first index to distance
                    reset running totals, including current values
        """

        sat_lat = l1b["sat_lat"][:].data
        sat_lon = l1b["sat_lon"][:].data
        r_lats = (sat_lat * np.pi) / 180
        r_lons = (sat_lon * np.pi) / 180
        r_points = np.transpose([r_lats, r_lons])
        valid = l1b["valid"][:].data.flatten().astype(np.bool_)

        floe_chord_length = np.full_like(sat_lat, np.nan)

        first_floe_index = None
        last_floe_index = None

        for index in range(len(sat_lat)):
            if not valid[index]:
                continue

            if valid[index] and first_floe_index is None:
                first_floe_index = index
                continue

            distance = (
                haversine_distances(r_points[first_floe_index], r_points[index]) * self.earth_radius
            )

            if distance > self.max_distance:
                floe_length = (
                    haversine_distances(r_points[first_floe_index], r_points[last_floe_index])
                    * self.earth_radius
                )
                floe_chord_length[first_floe_index] = floe_length
                first_floe_index = index

            last_floe_index = index

        shared_dict["floe_chord_length"] = floe_chord_length

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
