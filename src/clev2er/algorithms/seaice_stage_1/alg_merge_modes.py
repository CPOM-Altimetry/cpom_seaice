"""clev2er.algorithms.seaice_stage_1.alg_merge_modes.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

This algorithm will add append the data from the current file to the matching merge file based on
orbit number. The merge file will contain all relevant data for each crossing of the Arctic for SAR
and SARIn modes.

#Main initialization (init() function) steps/resources required

Check that merge file location exists

#Main process() function steps

Get orbit number from Dataset (NC File)
If the merge file does not exist, create a new one
Append the data from the current file to the existing data within the merge file
Close merge file

#Main finalize() function steps

None

#Contribution to shared_dict

None (Chain ends)

#Requires from shared_dict

"block number"
"packet id"
"elevation"


Author: Ben Palmer
Date: 22 Jul 2024
"""

import os
from typing import Tuple

import numpy as np
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
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        self.merge_file_dir = self.config["alg_merge_modes"]["merge_file_dir"]

        if not (os.path.exists(self.merge_file_dir) and os.path.isdir(self.merge_file_dir)):
            raise FileNotFoundError("Specified merge file directory does not exist")

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

        # check all variables being deposited in the merge file are of equal length
        if not (
            shared_dict["block_number"].size
            == shared_dict["packet_count"].size
            == shared_dict["measurement_time"].size
            == shared_dict["sat_lat"].size
            == shared_dict["sat_lon"].size
            == shared_dict["elevation"].size
            == shared_dict["lead_floe_class"].size
        ):
            self.log.error("Variables that will be added to merge file are not of equal length")

            for var_name in [
                "block_number",
                "packet_count",
                "measurement_time",
                "sat_lat",
                "sat_lon",
                "elevation",
                "lead_floe_class",
            ]:
                self.log.error("   %s - size=%d", var_name, shared_dict[var_name].size)

            return (False, "VarLengthError")

        # Create output file locations
        output_file_name = f"merge_{l1b.rel_orbit_number:04d}.nc"
        output_file_path = os.path.join(self.merge_file_dir, output_file_name)

        # If output file does not already exist, create new file
        # Else, load up the output file
        if not os.path.exists(output_file_path):
            output_nc: Dataset = Dataset(output_file_path, mode="w")
            output_nc.createDimension("n_samples", None)

            output_nc.createVariable("packet_count", "i4", ("n_samples",), compression="zlib")
            output_nc.createVariable("block_number", "i4", ("n_samples",), compression="zlib")
            output_nc.createVariable("measurement_time", "f8", ("n_samples",), compression="zlib")
            output_nc.createVariable("sat_lat", "f4", ("n_samples",), compression="zlib")
            output_nc.createVariable("sat_lon", "f4", ("n_samples",), compression="zlib")
            output_nc.createVariable("elevation", "f4", ("n_samples",), compression="zlib")
            output_nc.createVariable("lead_floe_class", "f4", ("n_samples",), compression="zlib")
        else:
            output_nc = Dataset(output_file_path, mode="a")

        # append data from the current file to data within the merge file
        packet_count = np.concatenate((output_nc["packet_count"][:], shared_dict["packet_count"]))
        block_number = np.concatenate((output_nc["block_number"][:], shared_dict["block_number"]))
        measurement_time = np.concatenate(
            (output_nc["measurement_time"][:], shared_dict["measurement_time"])
        )
        sat_lat = np.concatenate((output_nc["sat_lat"][:], shared_dict["sat_lat"]))
        sat_lon = np.concatenate((output_nc["sat_lon"][:], shared_dict["sat_lon"]))
        elevation = np.concatenate((output_nc["elevation"][:], shared_dict["elevation"]))
        lead_floe_class = np.concatenate(
            (output_nc["lead_floe_class"][:], shared_dict["lead_floe_class"])
        )

        # add the data to the merge file
        output_nc["packet_count"][:] = packet_count
        output_nc["block_number"][:] = block_number
        output_nc["measurement_time"][:] = measurement_time
        output_nc["sat_lat"][:] = sat_lat
        output_nc["sat_lon"][:] = sat_lon
        output_nc["elevation"][:] = elevation
        output_nc["lead_floe_class"][:] = lead_floe_class

        # close file
        output_nc.close()

        self.log.info("Appended data to %s", output_file_name)
        # -------------------------------------------------------------------
        # Returns (True,'') if successful
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
