"""clev2er.algorithms.seaice_stage_1..py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

This algorithm will add append the data from the current file to the matching merge file based on
orbit number. The merge file will contain all relevant data for each crossing of the Arctic for SAR
and SARIn modes.

#Main initialization (init() function) steps/resources required

Check that merge file location exists

#Main process() function steps

Get the date of the arc from the first measurement
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

import fcntl
import os
import signal
from typing import Tuple

import numpy as np
from astropy.time import Time
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm


def timeout_handler(signum, frame):
    """Handler function to raise an Error when we timeout"""
    raise TimeoutError("Lock acquisition failed after waiting for timeout duration")


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

        self.merge_file_dir = self.config["alg_merge_months"]["merge_file_dir"]
        self.timeout = self.config["alg_merge_months"]["mp_file_timeout"]

        if not (os.path.exists(self.merge_file_dir) and os.path.isdir(self.merge_file_dir)):
            raise FileNotFoundError("Specified merge file directory does not exist")

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
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

        # check all variables being deposited in the merge file are of equal length
        if not (
            l1b["block_number"].size
            == l1b["packet_count"].size
            == l1b["measurement_time"].size
            == l1b["lead_floe_class"].size
            == l1b["sat_lat"].size
            == l1b["sat_lon"].size
            == shared_dict["thickness"].size
            == shared_dict["freeboard"].size
            == shared_dict["seaice_type"].size
        ):
            self.log.error("Variables that will be added to merge file are not of equal length")

            for var_name in [
                "block_number",
                "packet_count",
                "measurement_time",
                "sat_lat",
                "sat_lon",
            ]:
                self.log.error("   %s - size=%d", var_name, l1b[var_name].size)

            for var_name in ["thickness", "freeboard", "seaice_type"]:
                self.log.error("   %s - size=%d", var_name, shared_dict[var_name].size)

            return (False, "VarLengthError")

        packet_count = l1b["packet_count"][:].data
        block_number = l1b["block_number"][:].data
        measurement_time = l1b["measurement_time"][:].data
        sample_valid = shared_dict["valid"].astype(np.bool_)
        sat_lat = l1b["sat_lat"][:].data
        sat_lon = l1b["sat_lon"][:].data
        surface_type = l1b["lead_floe_class"][:].data
        seaice_conc = l1b["seaice_conc"][:].data
        thickness = shared_dict["thickness"]
        freeboard = shared_dict["freeboard_corr"]
        seaice_type = shared_dict["seaice_type"]

        # Create output file locations
        # Set up output file
        f_time = Time(np.min(l1b["measurement_time"]), format="unix_tai").strftime("%Y%m")

        # group files by year
        year_dir = os.path.join(self.merge_file_dir, f_time[:4])
        if not os.path.isdir(year_dir):
            os.mkdir(year_dir)

        merge_file_name = f"{f_time}_merge.nc"
        output_file_path = os.path.join(year_dir, merge_file_name)
        output_lock_file_path = os.path.join(year_dir, "." + merge_file_name + ".lock")

        # Acquire lock file to stop other processes from trying to add to the file
        with open(output_lock_file_path, "w", encoding="utf-8") as lock:
            # Handle timeouts if we fail to acquire the lock for a set period
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                signal.alarm(0)

                # If output file does not already exist, create new file
                # Else, load up the output file
                if not os.path.exists(output_file_path):
                    output_nc: Dataset = Dataset(output_file_path, mode="w")
                    output_nc.createDimension("n_samples", None)

                    output_nc.createVariable(
                        "packet_count", "i4", ("n_samples",), compression="zlib"
                    )
                    output_nc.createVariable(
                        "block_number", "i4", ("n_samples",), compression="zlib"
                    )
                    output_nc.createVariable(
                        "measurement_time", "f8", ("n_samples",), compression="zlib"
                    )
                    output_nc.createVariable("valid", "b", ("n_samples",), compression="zlib")

                    output_nc.createVariable("sat_lat", "f4", ("n_samples",), compression="zlib")
                    output_nc.createVariable("sat_lon", "f4", ("n_samples",), compression="zlib")
                    output_nc.createVariable(
                        "surface_type", "i4", ("n_samples",), compression="zlib"
                    )
                    output_nc.createVariable("thickness", "f4", ("n_samples",), compression="zlib")
                    output_nc.createVariable("freeboard", "f4", ("n_samples",), compression="zlib")
                    output_nc.createVariable(
                        "seaice_conc", "f4", ("n_samples",), compression="zlib"
                    )
                    output_nc.createVariable(
                        "seaice_type", "i4", ("n_samples",), compression="zlib"
                    )
                else:
                    output_nc = Dataset(output_file_path, mode="a")

                # append data from the current file to data within the merge file
                packet_count = np.concatenate((output_nc["packet_count"][:], packet_count))
                block_number = np.concatenate((output_nc["block_number"][:], block_number))
                measurement_time = np.concatenate(
                    (output_nc["measurement_time"][:], measurement_time)
                )
                sample_valid = np.concatenate((output_nc["valid"][:], sample_valid))
                sat_lat = np.concatenate((output_nc["sat_lat"][:], sat_lat))
                sat_lon = np.concatenate((output_nc["sat_lon"][:], sat_lon))
                surface_type = np.concatenate((output_nc["lead_floe_class"][:], surface_type))
                thickness = np.concatenate((output_nc["thickness"][:], thickness))
                freeboard = np.concatenate((output_nc["freeboard"][:], freeboard))
                seaice_conc = np.concatenate((output_nc["seaice_conc"][:], seaice_conc))
                seaice_type = np.concatenate((output_nc["seaice_type"][:], seaice_type))

                # add the data to the merge file
                output_nc["packet_count"][:] = packet_count
                output_nc["block_number"][:] = block_number
                output_nc["measurement_time"][:] = measurement_time
                output_nc["valid"][:] = sample_valid
                output_nc["sat_lat"][:] = sat_lat
                output_nc["sat_lon"][:] = sat_lon
                output_nc["surface_type"][:] = surface_type
                output_nc["thickness"][:] = thickness
                output_nc["freeboard"][:] = freeboard
                output_nc["seaice_conc"][:] = seaice_conc
                output_nc["seaice_type"][:] = seaice_type

                # close file
                output_nc.close()
            finally:
                # release lock when we're done with the file
                signal.alarm(0)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        self.log.info("Appended data to %s", output_file_path)
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
