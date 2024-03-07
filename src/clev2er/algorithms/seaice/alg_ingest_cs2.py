""" clev2er.algorithms.seaice.alg_ingest_cs2.py

    Algorithm class module, used to implement a single chain algorithm

    #Description of this Algorithm's purpose
    
    To ingest CryoSat L1b files and extract the necessary parameters to a common
    set of parameter names
    
    #Main initialization (init() function) steps/resources required

    None

    #Main process() function steps

    Read 20Hz variables to memory
    lat_20_ku -> sat_lat
    lon_20_ku -> sat_lon
    time_20_ku --> measurement_time
    alt_20_ku --> sat_altitude
    window_delay --> window_delay
    pwr_waveform_20_ku --> waveform
    stack_std_20_ku --> waveform_ssd
    flag_mcd_20_ku --> mcd_flag
    
    Read 1Hz variables to memory and extrapolate to 20Hz 
    mod_dry_tropo_cor_01 --> dry_trop_correction
    mod_wet_tropo_cor_01 --> wet_trop_correction
    iono_cor_01 --> inv_baro_correction
    inv_bar_cor_01 --> iono_correction
    ocean_tide_01 --> ocean_tide
    ocean_tide_eq_01 --> long_period_tide
    load_tide_01 --> loading_tide
    solid_earth_tide_01 --> earth_tide
    pole_tide_01 --> pole_tide
    surf_type_01 --> surface_type


    #Contribution to shared_dict

    shared_dict["sat_lat"] (np.array[int]) : latitude of measurements in degs N (-90,90)
    shared_dict["sat_lon"] (np.array[int]) : longitude of measurements in degs E (0..360)
    shared_dict["sat_altitude"] (np.array[int]) : array of reading altitudes
    shared_dict["measurement_time"] (np.array[int]) : array of reading times
    shared_dict["window_delay"] (np.array[int]) : array of window delays
    shared_dict["waveform"] (np.array[int]) : array of waveform power samples
    shared_dict["waveform_ssd"] (np.array[int]) : array of stack standard devations 
                                                for each waveform
    shared_dict["dry_trop_correction"] (np.array[int]) : array of dry tropospheric corrections
    shared_dict["wet_trop_correction"] (np.array[int]) : array of wet tropospheric corrections
    shared_dict["inv_baro_correction"] (np.array[int]) : array of inverse barometer corrections
    shared_dict["iono_correction"] (np.array[int]) : array of ionospheric corrections
    shared_dict["ocean_tide"] (np.array[int]) : array of ocean tides
    shared_dict["long_period_tide"] (np.array[int]) : array of long period tides
    shared_dict["loading_tide"] (np.array[int]) : array of loading tides
    shared_dict["earth_tide"] (np.array[int]) : array of solid earth tides
    shared_dict["pole_tide"] (np.array[int]) : array of pole tides
    shared_dict["surface_type"] (np.array[int]) : array of surface type flags

    #Requires from shared_dict

    None
"""

from datetime import datetime
from functools import partial
from typing import Tuple

from codetiming import Timer  # used to time the Algorithm.process() function
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.io.ingest_l1b import unpack

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

        self.time_20_hz = self.config["alg_ingest_cs2"]["time_var_20Hz"]
        self.time_01_hz = self.config["alg_ingest_cs2"]["time_var_01Hz"]

        self.unpack = partial(unpack, time_var_1=self.time_20_hz, time_var_2=self.time_01_hz)

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

        # self.log.debug("example of a debug message")
        # self.log.info("example of an info message")
        # self.log.error("example of an error message")

        try:
            if "SARIN" in l1b.sir_op_mode:
                shared_dict["instr_mode"] = "SIN"
            elif "SAR" in l1b.sir_op_mode:
                shared_dict["instr_mode"] = "SAR"
            else:
                return (False, f"Invalid mode attribute .sir_op_mode in L1b file {l1b.sir_op_mode}")
        except AttributeError:
            return (False, "Missing attribute .sir_op_mode in L1b file")

        self.log.info("Found file operating mode - %s", shared_dict["instr_mode"])

        # Read L1b variables and store in the shared dictionary using common names
        # 20 Hz variables
        shared_dict["sat_lat"] = self.unpack("lat_20_ku", l1b)
        # convert longitude to 0..360 (from -180,180)
        shared_dict["sat_lon"] = self.unpack("lon_20_ku", l1b) % 360.0  #
        # timestamps in file are seconds from 1/1/2000, convert to seconds from 1/1/1970
        shared_dict["measurement_time"] = (
            self.unpack("time_20_ku", l1b) + datetime(2000, 1, 1).timestamp()
        )
        shared_dict["sat_altitude"] = self.unpack("alt_20_ku", l1b)
        shared_dict["window_delay"] = self.unpack("window_del_20_ku", l1b)
        shared_dict["waveform"] = self.unpack("pwr_waveform_20_ku", l1b)
        shared_dict["waveform_ssd"] = self.unpack("stack_std_20_ku", l1b)
        shared_dict["mcd_flag"] = self.unpack("flag_mcd_20_ku", l1b)

        # 1 Hz variables
        shared_dict["dry_trop_correction"] = self.unpack("mod_dry_tropo_cor_01", l1b)
        shared_dict["wet_trop_correction"] = self.unpack("mod_wet_tropo_cor_01", l1b)
        shared_dict["iono_correction"] = self.unpack("iono_cor_01", l1b)
        shared_dict["inv_baro_correction"] = self.unpack("inv_bar_cor_01", l1b)
        shared_dict["ocean_tide"] = self.unpack("ocean_tide_01", l1b)
        shared_dict["long_period_tide"] = self.unpack("ocean_tide_eq_01", l1b)
        shared_dict["loading_tide"] = self.unpack("load_tide_01", l1b)
        shared_dict["earth_tide"] = self.unpack("solid_earth_tide_01", l1b)
        shared_dict["pole_tide"] = self.unpack("pole_tide_01", l1b)
        shared_dict["surface_type"] = self.unpack("surf_type_01", l1b)

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
