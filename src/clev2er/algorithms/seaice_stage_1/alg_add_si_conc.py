"""clev2er.algorithms.seaice_stage_1_stage_1.alg_add_si_conc.py

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
from datetime import datetime, timedelta
from typing import Dict, Literal, Tuple

import numpy as np
import pyproj as proj
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from pyproj import Transformer
from scipy.spatial import cKDTree

from clev2er.algorithms.base.base_alg import BaseAlgorithm


def _find_single_conc_file(glob_pattern, hemisphere: Literal["north", "south"]):
    file_path: str | None = None

    # Find files that match the date
    file_paths = glob.glob(glob_pattern)

    file_paths = _filter_files_by_hemisphere(file_paths, hemisphere)

    # part 1: try nc first, if not then try dat
    # now it's more complicated. check which datasets we're using
    nc_paths = [x for x in file_paths if x.endswith(".nc")]
    if len(nc_paths) > 1:
        raise RuntimeError(f"Found too many nc concentration files that matched {glob_pattern}")
    if len(nc_paths) == 1:
        file_path = nc_paths[0]

    # if a netcdf hasn't been found, try to find a .dat file
    if file_path is None:
        dat_paths = [x for x in file_paths if x.endswith(".dat")]
        if len(dat_paths) > 1:
            raise RuntimeError(
                f"Found too many dat concentration files that matched {glob_pattern}"
            )

        if len(dat_paths) == 1:
            file_path = dat_paths[0]

    # if a file has still not been found, raise an error
    if file_path is None:
        raise FileNotFoundError(f"Cannot find nc or dat concentration file matching {glob_pattern}")

    return file_path


def _filter_files_by_hemisphere(
    file_list: list[str], hemisphere: Literal["north", "south"]
) -> list[str]:
    if hemisphere == "north":
        return [f for f in file_list if "_nh_" in f or "_N" in f]

    return [f for f in file_list if "_sh_" in f or "_S" in f]


class Algorithm(BaseAlgorithm):
    # pylint: disable=too-many-instance-attributes
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

        self.hemi = self.config["shared"]["hemisphere"]
        self.fill_conc = self.config["alg_add_si_conc"]["fill_conc"]
        self.fill_lat_threshold = self.config["alg_add_si_conc"]["fill_lat_threshold"]

        # since we use different datasets for conc depending on the date...
        # use config to decide which we run and for which dates
        # NOTE: OSISAF files are all for .nc reading only, NSIDC can be .nc or .dat

        # NSIDC SSMI (default for 2010-2024)
        self.use_nsidc_ssmi = (
            "nsidc_ssmi" in self.config["alg_add_si_conc"]["datasets"]
            and self.config["alg_add_si_conc"]["datasets"]["nsidc_ssmi"]["active"]
        )
        if self.use_nsidc_ssmi:
            self.nsidc_ssmi_dir = self.config["alg_add_si_conc"]["datasets"]["nsidc_ssmi"][
                "base_dir"
            ]
            self.nsidc_ssmi_start_time = datetime.strptime(
                self.config["alg_add_si_conc"]["datasets"]["nsidc_ssmi"]["start_date"], "%d-%m-%Y"
            )
            self.nsidc_ssmi_end_time = (
                datetime.strptime(
                    self.config["alg_add_si_conc"]["datasets"]["nsidc_ssmi"]["end_date"], "%d-%m-%Y"
                )
                + timedelta(days=1, seconds=-1)
                if self.config["alg_add_si_conc"]["datasets"]["nsidc_ssmi"]["end_date"] is not None
                else None
            )
            self.log.info(
                "Using NSIDC SSMI dataset from %s to %s",
                self.nsidc_ssmi_start_time,
                self.nsidc_ssmi_end_time,
            )

        # OSISAF SSMI (used for 2025-onwards)
        self.use_osisaf_ssmi = (
            "osisaf_ssmi" in self.config["alg_add_si_conc"]["datasets"]
            and self.config["alg_add_si_conc"]["datasets"]["osisaf_ssmi"]["active"]
        )
        if self.use_osisaf_ssmi:
            self.osisaf_ssmi_dir = self.config["alg_add_si_conc"]["datasets"]["osisaf_ssmi"][
                "base_dir"
            ]
            self.osisaf_ssmi_start_time = datetime.strptime(
                self.config["alg_add_si_conc"]["datasets"]["osisaf_ssmi"]["start_date"], "%d-%m-%Y"
            )
            self.osisaf_ssmi_end_time = (
                datetime.strptime(
                    self.config["alg_add_si_conc"]["datasets"]["osisaf_ssmi"]["end_date"],
                    "%d-%m-%Y",
                )
                - timedelta(seconds=1)
                if self.config["alg_add_si_conc"]["datasets"]["osisaf_ssmi"]["end_date"] is not None
                else None
            )
            self.log.info(
                "Using OSISAF SSMI dataset from %s to %s",
                self.osisaf_ssmi_start_time,
                self.osisaf_ssmi_end_time,
            )

        # OSISAF ASMR-2 (not used currently, adding as a placeholder)
        self.use_osisaf_asmr2 = (
            "osisaf_asmr2" in self.config["alg_add_si_conc"]["datasets"]
            and self.config["alg_add_si_conc"]["datasets"]["osisaf_asmr2"]["active"]
        )
        if self.use_osisaf_asmr2:
            self.osisaf_asmr2_dir = self.config["alg_add_si_conc"]["datasets"]["osisaf_asmr2"][
                "base_dir"
            ]
            self.osisaf_asmr2_start_time = datetime.strptime(
                self.config["alg_add_si_conc"]["datasets"]["osisaf_asmr2"]["start_date"], "%d-%m-%Y"
            )
            self.osisaf_asmr2_end_time = (
                datetime.strptime(
                    self.config["alg_add_si_conc"]["datasets"]["osisaf_asmr2"]["end_date"],
                    "%d-%m-%Y",
                )
                + timedelta(days=1, seconds=-1)
                if self.config["alg_add_si_conc"]["datasets"]["osisaf_asmr2"]["end_date"]
                is not None
                else None
            )
            self.log.info(
                "Using OSISAF ASMR-2 dataset from %s to %s",
                self.osisaf_asmr2_start_time,
                self.osisaf_asmr2_end_time,
            )

        input_projection = self.config["alg_add_si_conc"]["input_projection"]
        output_projection = self.config["shared"]["output_projection"]

        self.log.info(
            "Transforming projection from %s to %s for value reading",
            input_projection,
            output_projection,
        )

        crs_input = proj.Proj(input_projection)
        crs_output = proj.Proj(output_projection)
        self.lonlat_to_xy = Transformer.from_proj(crs_input, crs_output, always_xy=True)
        self.xy_to_lonlat = Transformer.from_proj(crs_output, crs_input, always_xy=True)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=too-many-branches
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
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        si_concentration = np.zeros(shared_dict["sat_lat"].size) * np.nan

        # for each timestamp, lat and lon in shared memory:
        for wv_num, (wv_timestamp, wv_lat, wv_lon) in enumerate(
            zip(shared_dict["measurement_time"], shared_dict["sat_lat"], shared_dict["sat_lon"])
        ):
            wv_datetime = datetime.fromtimestamp(wv_timestamp)
            file_date = wv_datetime.strftime("%Y%m%d")

            if self.most_recent_file["date"] == file_date:
                # If date is the same as the most recent file date, get values from dict
                file_point_tree = self.most_recent_file["tree"]
                file_values = self.most_recent_file["values"]

            else:
                # Else, read the file, create the KDTree and store the values
                # in most recent file dict for later use
                self.log.info("Loading new concentration data file - %s", file_date)

                try:
                    if (
                        self.use_nsidc_ssmi
                        and self.nsidc_ssmi_start_time <= wv_datetime
                        and (
                            self.nsidc_ssmi_end_time is None
                            or wv_datetime <= self.nsidc_ssmi_end_time
                        )
                    ):
                        self.log.debug("Using NSIDC SSMI")
                        file_path = _find_single_conc_file(
                            os.path.join(self.nsidc_ssmi_dir, file_date[:4], f"*{file_date}*"),
                            self.hemi,
                        )

                        # if a file has been found, check the extension and load it
                        if file_path.endswith(".dat"):
                            self.log.debug("Using .dat file")
                            # Read the dat file
                            sea_ice_conc = np.transpose(np.genfromtxt(file_path))
                            file_lats = sea_ice_conc[2]
                            # convert to 0..360 to match shared_dict values
                            file_lons = sea_ice_conc[3] % 360.0
                            file_values = sea_ice_conc[4]
                            file_values[
                                file_values == -999.0
                            ] = np.nan  # Turn -999.0 values to NaNs
                            file_x, file_y = self.lonlat_to_xy.transform(file_lons, file_lats)

                        elif file_path.endswith(".nc"):
                            self.log.debug("Using .nc file")
                            with Dataset(file_path, mode="r") as nc:
                                file_values_frac = nc["F18_ICECON"][:].data.flatten()
                                file_x_1d = nc["x"][:].data
                                file_y_1d = nc["y"][:].data

                    elif (
                        self.use_osisaf_ssmi
                        and self.osisaf_ssmi_start_time <= wv_datetime
                        and (
                            self.osisaf_ssmi_end_time is None
                            or wv_datetime <= self.osisaf_ssmi_end_time
                        )
                    ):
                        self.log.debug("Using OSISAF SSMI data")
                        file_path = _find_single_conc_file(
                            os.path.join(self.osisaf_ssmi_dir, file_date[:4], f"*{file_date}*"),
                            self.hemi,
                        )

                        with Dataset(file_path, mode="r") as nc:
                            file_values_frac = nc["ice_conc"][:].data.flatten()
                            file_x_1d = nc["xc"][:].data
                            file_y_1d = nc["yc"][:].data

                    elif (
                        self.use_osisaf_asmr2
                        and self.osisaf_asmr2_start_time <= wv_datetime
                        and (
                            self.osisaf_asmr2_end_time is None
                            or wv_datetime <= self.osisaf_asmr2_end_time
                        )
                    ):
                        self.log.debug("Using OSISAF ASMR-2 data")
                        file_path = _find_single_conc_file(
                            os.path.join(self.osisaf_asmr2_dir, file_date[:4], f"*{file_date}*"),
                            self.hemi,
                        )

                        with Dataset(file_path, mode="r") as nc:
                            file_values_frac = nc["ice_conc"][:].data.flatten()
                            file_x_1d = nc["xc"][:].data
                            file_y_1d = nc["yc"][:].data

                    else:
                        raise RuntimeError(
                            "No suitable dataset found! Aglorithm requires at least one dataset"
                            " active and to have a suitable date"
                        )
                except FileNotFoundError:
                    self.log.error("Cannot find file for %s", file_date)
                    return (False, "SKIP_OK")

                self.log.info("Found file %s", file_path)

                if file_path.endswith(".nc"):
                    # file values are read in as fractions (0->1) with flags for invalid values
                    # convert all flags to nan
                    file_values_frac[file_values_frac > 1] = np.nan
                    # convert fraction to percentage
                    file_values = np.round((file_values_frac * 100), decimals=4)
                    # x and y need to be in equal length to values
                    file_x, file_y = np.meshgrid(file_x_1d, file_y_1d)
                    file_x = file_x.flatten()
                    file_y = file_y.flatten()
                    # need lat if using filling method below
                    _, file_lats = self.xy_to_lonlat.transform(file_x, file_y)

                # part 3: fill above a latitude threshold if set (arctic only)
                if self.fill_conc:
                    # Fill NaN values above lat threshold to mean of all lats above threshold
                    lats_above_threshold = (
                        file_lats > self.fill_lat_threshold
                    )  # get points above threshold
                    values_to_fill: np.ndarray = np.isnan(file_values)  # get unknowns
                    fill_value = np.max(
                        (np.mean(file_values[lats_above_threshold & ~values_to_fill]), 0)
                    )  # get mean of known above threshold
                    if np.isnan(fill_value):
                        raise RuntimeError("Calculated fill value is NaN")
                    self.log.info("Filling concentrations using mean value - %0.3f", fill_value)
                    file_values[lats_above_threshold & values_to_fill] = fill_value

                # Convert the longitudes and latitudes to (x, y) pairs and create a KDTree of points
                file_points = np.transpose((file_x, file_y))
                file_point_tree = cKDTree(file_points)

                # Save the loaded date, KDTree, and values
                # Faster to save the tree than save the lon + lat values and recreate
                # the tree every time

                self.most_recent_file["date"] = file_date
                self.most_recent_file["tree"] = file_point_tree
                self.most_recent_file["values"] = file_values

            wv_x, wv_y = self.lonlat_to_xy.transform(wv_lon, wv_lat)

            file_neighbouring_dist, file_neighbouring_indices = file_point_tree.query(
                (wv_x, wv_y), k=1, distance_upper_bound=18000
            )
            if not np.isfinite(file_neighbouring_dist):
                si_concentration[wv_num] = 0
            else:
                si_concentration[wv_num] = file_values[file_neighbouring_indices]

        self.log.info("NaNs in concentration array - %d", sum(np.isnan(si_concentration)))
        if all(np.isnan(si_concentration)):
            self.log.warning("ALL CONCENTRATIONS ARE NaN")
        else:
            self.log.info(
                "Sea ice concentration: Max=%f Mean=%f Min=%f Zeroes=%d",
                np.nanmax(si_concentration),
                np.nanmean(si_concentration),
                np.nanmin(si_concentration),
                sum(si_concentration <= 0.0),
            )

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
