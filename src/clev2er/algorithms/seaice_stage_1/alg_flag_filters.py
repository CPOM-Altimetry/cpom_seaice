"""clev2er.algorithms.seaice_stage_1.alg_flag_filters.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Filter records by discarding records which do not have the required flags:
    least significant bit of measurement flag = 0
    surface type flag = 0

#Main initialization (init() function) steps/resources required

None

#Main process() function steps

Find index of mcd_flag array where value of least significant bit is 0
Find the index of surface_type array where value is 0.
Use the index to filter all arrays to remove unwanted records.

#Contribution to shared_dict

shared_dict["flag_index"] (np.array[int]) : indices of area-filtered arrays that have correct
                                            mcd and surface type flags

#Requires from shared_dict

shared_dict["sat_lat"]
shared_dict["sat_lon"]
shared_dict["sat_altitude"]
shared_dict["measurement_time"]
shared_dict["window_del_20_ku"]
shared_dict["waveform"]
shared_dict["waveform_ssd"]
shared_dict["dry_trop_correction"]
shared_dict["wet_trop_correction"]
shared_dict["inv_baro_correction"]
shared_dict["iono_correction"]
shared_dict["ocean_tide"]
shared_dict["long_period_tide"]
shared_dict["loading_tide"]
shared_dict["earth_tide"]
shared_dict["pole_tide"]
shared_dict["surface_type"]
shared_dict["mcd_flag"]
"""

from typing import Tuple

import numpy as np
from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.io.binary_utils import check_bit, to_binary

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

        self.binary_width = self.config["alg_flag_filters"]["binary_width"]
        self.surf_ocean_flag = self.config["alg_flag_filters"]["surf_ocean_flag"]

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

        total_points = shared_dict["sat_lat"].size

        # filter by mcd_flag
        # convert mcd flag from integer to binary
        flag_binaries = to_binary(shared_dict["mcd_flag"], width=self.binary_width)

        # check bits at 1, 6, 7, 18, 22, 25, 26, 27, 28 (Not array indexes, so -1 for each)
        bit_1 = check_bit(flag_binaries, 1)
        bit_6 = check_bit(flag_binaries, 6)
        bit_7 = check_bit(flag_binaries, 7)
        bit_18 = check_bit(flag_binaries, 18)
        bit_22 = check_bit(flag_binaries, 22)
        bit_25 = check_bit(flag_binaries, 25)
        bit_26 = check_bit(flag_binaries, 26)
        bit_27 = check_bit(flag_binaries, 27)
        bit_28 = check_bit(flag_binaries, 28)

        mcd_index = ~(bit_1 | bit_6 | bit_7 | bit_18 | bit_22 | bit_25 | bit_26 | bit_27 | bit_28)
        num_confident = sum(mcd_index)  # find number of True values

        self.log.info(
            "Number of unconfident points = %d of %d",
            num_confident - total_points,
            total_points,
        )

        if num_confident == 0:
            self.log.info("No confident samples.")
            return (False, "SKIP_OK")

        # filter by surface type
        # make a boolean index so we can combine later
        ocean_index = shared_dict["surface_type"] == self.surf_ocean_flag
        num_ocean = sum(ocean_index)  # find number of True values

        self.log.info("Number of ocean points = %d of %d", num_ocean, total_points)

        if num_ocean == 0:
            self.log.info("No samples from ocean surfaces.")
            return (False, "SKIP_OK")

        # combine boolean indexes to int index
        combined_filter = np.where(ocean_index & mcd_index)[0]

        # Outputs of the algorithm saved to the shared_dict

        shared_dict["num_points_ocean"] = num_ocean
        shared_dict["indices_flags"] = combined_filter

        # filter the input parameter based on the area indices inside
        shared_dict["sat_lat"] = shared_dict["sat_lat"][combined_filter]
        shared_dict["sat_lon"] = shared_dict["sat_lon"][combined_filter]
        shared_dict["measurement_time"] = shared_dict["measurement_time"][combined_filter]
        shared_dict["andy_time"] = shared_dict["andy_time"][combined_filter]
        shared_dict["block_number"] = shared_dict["block_number"][combined_filter]
        shared_dict["packet_count"] = shared_dict["packet_count"][combined_filter]
        shared_dict["sat_altitude"] = shared_dict["sat_altitude"][combined_filter]
        shared_dict["window_delay"] = shared_dict["window_delay"][combined_filter]
        shared_dict["waveform"] = shared_dict["waveform"][combined_filter]
        shared_dict["waveform_ssd"] = shared_dict["waveform_ssd"][combined_filter]
        shared_dict["dry_trop_correction"] = shared_dict["dry_trop_correction"][combined_filter]
        shared_dict["wet_trop_correction"] = shared_dict["wet_trop_correction"][combined_filter]
        shared_dict["inv_baro_correction"] = shared_dict["inv_baro_correction"][combined_filter]
        shared_dict["iono_correction"] = shared_dict["iono_correction"][combined_filter]
        shared_dict["ocean_tide"] = shared_dict["ocean_tide"][combined_filter]
        shared_dict["long_period_tide"] = shared_dict["long_period_tide"][combined_filter]
        shared_dict["loading_tide"] = shared_dict["loading_tide"][combined_filter]
        shared_dict["earth_tide"] = shared_dict["earth_tide"][combined_filter]
        shared_dict["pole_tide"] = shared_dict["pole_tide"][combined_filter]
        shared_dict["surface_type"] = shared_dict["surface_type"][combined_filter]
        shared_dict["mcd_flag"] = shared_dict["mcd_flag"][combined_filter]

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

        # none required in this algorithm

        # ---------------------------------------------------------------------
