"""
Slope correction/geolocation function using an adpated  Roemer/LEPTA method
"""

import logging
from typing import Tuple

import numpy as np
import pyproj
from cpom.dems.dems import Dem
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
from pyproj import Transformer
from scipy.ndimage import median_filter

from clev2er.utils.cs2.geolocate.lrm_slope import slope_doppler

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

log = logging.getLogger(__name__)


def calculate_distances(
    x1_coord: float,
    y1_coord: float,
    z1_coord: float,
    x2_array: np.ndarray[float],
    y2_array: np.ndarray[float],
    z2_array: np.ndarray[float],
    squared_only=False,
) -> list[float]:
    """calculates the distances between a  refernce cartesian point (x1,y1,z1) in 3d space
    and a list of other points : x2[],y2[],z2[]

    Args:
        x1_coord (float): x coordinate of ref point
        y1_coord (float): y coordinate of ref point
        z1_coord (float): z coordinate of ref point
        x2_array (list[float]): list of x coordinates
        y2_array (list[float]): list of y coordinates
        z2_array (list[float]): list of z coordinates
        squared_only (bool) : if True, only calculate the squares of diffs and not sqrt
                              this will be faster, but doesn't give actual distances

    Returns:
        list[float]: list of distances between points x1,y1,z1 and x2[],y2[],z2[]
    """

    x2_array = np.array(x2_array)
    y2_array = np.array(y2_array)
    z2_array = np.array(z2_array)

    distances = (
        (x2_array - x1_coord) ** 2
        + (y2_array - y1_coord) ** 2
        + (z2_array - z1_coord) ** 2
    )

    if not squared_only:
        distances = np.sqrt(distances)

    return distances.tolist()  # Convert back to a regular Python list


def geolocate_lepta(
    l1b: Dataset,
    thisdem: Dem,
    config: dict,
    surface_type_20_ku: np.ndarray,
    geo_corrected_tracker_range: np.ndarray,
    retracker_correction: np.ndarray,
    leading_edge_start: np.ndarray,
    leading_edge_stop: np.ndarray,
    waveforms_to_include: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        l1b (Dataset): NetCDF Dataset of L1b file
        thisdem (Dem): Dem object used for Roemer/LEPTA correction
        config (dict): config dictionary containing ["lrm_lepta_geolocation"][params]
        surface_type_20_ku (np.ndarray): surface type for track, where 1 == grounded_ice
        geo_corrected_tracker_range (np.ndarray) : geo-corrected tracker range (NOT retracked)
        retracker_correction (np.ndarray) : retracker correction to range (m)
        leading_edge_start (np.ndarray) : position of start of waveform leading edge (decimal bins)
        leading_edge_stop (np.ndarray) : position of end of waveform leading edge (decimal bins)
        waveforms_to_include (np.ndarray) : boolean array of waveforms to include (False == reject)
    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        (height_20_ku, lat_poca_20_ku, lon_poca_20_ku, slope_ok)
    """
    # ------------------------------------------------------------------------------------
    # Get configuration parameters
    # ------------------------------------------------------------------------------------

    reference_bin_index = config["instrument"]["ref_bin_index_lrm"]
    range_bin_size = config["instrument"]["range_bin_size_lrm"]  # meters
    num_bins = config["instrument"]["num_range_bins_lrm"]
    across_track_beam_width = config["instrument"][
        "across_track_beam_width_lrm"
    ]  # meters

    range_increment = config["lrm_lepta_geolocation"]["range_increment"]
    max_increments = config["lrm_lepta_geolocation"]["num_range_increments"]

    # ------------------------------------------------------------------------------------

    # Get nadir latitude, longitude and satellite altitude from L1b
    lat_20_ku = l1b["lat_20_ku"][:].data
    lon_20_ku = l1b["lon_20_ku"][:].data % 360.0
    altitudes = l1b["alt_20_ku"][:].data

    # Transform to X,Y locs in DEM projection
    nadir_x, nadir_y = thisdem.lonlat_to_xy_transformer.transform(
        lon_20_ku, lat_20_ku
    )  # pylint: disable=unpacking-non-sequence

    # Interpolate DEM heights at nadir locations
    heights_at_nadir = thisdem.interp_dem(nadir_x, nadir_y)

    # Create working parameter arrays
    poca_x = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    poca_y = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    poca_z = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    slope_correction = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    slope_ok = np.full_like(nadir_x, dtype=bool, fill_value=True)
    height_20_ku = np.full_like(nadir_x, dtype=float, fill_value=np.nan)

    # ------------------------------------------------------------------------------------
    #  Loop through each track record
    # ------------------------------------------------------------------------------------

    for i, _ in enumerate(nadir_x):
        # By default, set POCA x,y to nadir, and height to Nan
        poca_x[i] = nadir_x[i]
        poca_y[i] = nadir_y[i]
        poca_z[i] = np.nan

        # if record is excluded due to previous checks, then skip
        if not waveforms_to_include[i]:
            continue

        # get the rectangular bounds about the track, adjusted for across track beam width and
        # the dem posting
        x_min = nadir_x[i] - (across_track_beam_width / 2 + thisdem.binsize)
        x_max = nadir_x[i] + (across_track_beam_width / 2 + thisdem.binsize)
        y_min = nadir_y[i] - (across_track_beam_width / 2 + thisdem.binsize)
        y_max = nadir_y[i] + (across_track_beam_width / 2 + thisdem.binsize)

        segment = [(x_min, x_max), (y_min, y_max)]

        # Extract the rectangular segment from the DEM
        try:
            xdem, ydem, zdem = thisdem.get_segment(segment, grid_xy=True, flatten=False)
        except (IndexError, ValueError, TypeError, AttributeError, MemoryError):
            slope_ok[i] = False
            continue
        except Exception:  # pylint: disable=W0718
            slope_ok[i] = False
            continue

        if config["lrm_lepta_geolocation"]["median_filter"]:
            smoothed_zdem = median_filter(zdem, size=3)
        else:
            smoothed_zdem = zdem

        xdem = xdem.flatten()
        ydem = ydem.flatten()
        zdem = smoothed_zdem.flatten()

        # Compute distance between each dem location and nadir in (x,y,z)
        dem_to_nadir_dists = calculate_distances(
            nadir_x[i], nadir_y[i], heights_at_nadir[i], xdem, ydem, zdem
        )

        # find where dem_to_nadir_dists is within beam. ie extract circular area
        include_dem_indices = np.where(
            np.array(dem_to_nadir_dists) < (across_track_beam_width / 2.0)
        )[0]
        if len(include_dem_indices) == 0:
            slope_ok[i] = False
            continue

        xdem = xdem[include_dem_indices]
        ydem = ydem[include_dem_indices]
        zdem = zdem[include_dem_indices]

        # Check DEM segment for bad values and remove
        nan_mask = np.isnan(zdem)
        include_only_good_zdem_indices = np.where(~nan_mask)[0]
        if len(include_only_good_zdem_indices) < 1:
            slope_ok[i] = False
            continue

        xdem = xdem[include_only_good_zdem_indices]
        ydem = ydem[include_only_good_zdem_indices]
        zdem = zdem[include_only_good_zdem_indices]

        # Only keep DEM heights which are in a sensible range
        # this step removes DEM values set to fill_value (a high number)
        valid_dem_heights = np.where(zdem < 5000.0)[0]
        if len(valid_dem_heights) < 1:
            slope_ok[i] = False
            continue

        xdem = xdem[valid_dem_heights]
        ydem = ydem[valid_dem_heights]
        zdem = zdem[valid_dem_heights]

        # Compute distance between each remaining dem location and satellite
        dem_to_sat_dists = calculate_distances(
            nadir_x[i], nadir_y[i], altitudes[i], xdem, ydem, zdem
        )

        # Optionally limit DEM points further by restricting to points where their range to
        # satellite is within the trackers range window

        if config["lrm_lepta_geolocation"]["filter_on_tracker_range_window"]:
            range_to_window_start = (
                geo_corrected_tracker_range[i] - (reference_bin_index) * range_bin_size
            )
            # calculate distance from sat to bottom of range window
            range_to_window_end = (
                geo_corrected_tracker_range[i]
                + (num_bins - reference_bin_index) * range_bin_size
            )

            indices_within_range_window = np.where(
                np.logical_and(
                    dem_to_sat_dists >= range_to_window_start,
                    dem_to_sat_dists <= range_to_window_end,
                )
            )[0]
            if len(indices_within_range_window) == 0:
                slope_ok[i] = False
                continue
            dem_to_sat_dists = np.array(dem_to_sat_dists)[indices_within_range_window]
            xdem = xdem[indices_within_range_window]
            ydem = ydem[indices_within_range_window]
            zdem = zdem[indices_within_range_window]

        # Optionally limit DEM points further by restricting to points where their range to
        # satellite is within the Leading Edge min/max range
        #   - if no points found, incrementally expand window up to the tracker range window

        if config["lrm_lepta_geolocation"]["filter_on_leading_edge"]:
            range_to_window_start = (
                geo_corrected_tracker_range[i] - (reference_bin_index) * range_bin_size
            )
            range_to_window_end = (
                geo_corrected_tracker_range[i]
                + (num_bins - reference_bin_index) * range_bin_size
            )

            range_to_le_start = (
                geo_corrected_tracker_range[i]
                - (reference_bin_index - leading_edge_start[i][0]) * range_bin_size
            )
            range_to_le_end = (
                geo_corrected_tracker_range[i]
                + (leading_edge_stop[i][0] - reference_bin_index) * range_bin_size
            )

            if config["lrm_lepta_geolocation"]["use_bottom_half_of_leading_edge"]:
                le_width = range_to_le_end - range_to_le_start

                # Try bottom quater of LE first
                indices_within_range_window = np.where(
                    np.logical_and(
                        dem_to_sat_dists >= range_to_le_start,
                        dem_to_sat_dists <= range_to_le_start + le_width / 4.0,
                    )
                )[0]

                # Try all of LE if no DEM points match bottom half
                if len(indices_within_range_window) == 0:
                    # log.debug("No points in bottom half")
                    indices_within_range_window = np.where(
                        np.logical_and(
                            dem_to_sat_dists >= range_to_le_start,
                            dem_to_sat_dists <= range_to_le_end,
                        )
                    )[0]
            else:
                indices_within_range_window = np.where(
                    np.logical_and(
                        dem_to_sat_dists >= range_to_le_start,
                        dem_to_sat_dists <= range_to_le_end,
                    )
                )[0]

            window_start_reached = False
            window_end_reached = False
            num_increments = 0

            while len(indices_within_range_window) == 0:
                log.debug(
                    "No matching dem points in LE, num_increments=%d ", num_increments
                )

                # Incrementally expand range window by
                # config["lrm_lepta_geolocation"]["range_increment"]
                range_to_le_start -= range_increment
                range_to_le_end += range_increment
                if range_to_le_start <= range_to_window_start:
                    window_start_reached = True
                    range_to_le_start = range_to_window_start
                if range_to_le_end >= range_to_window_end:
                    range_to_le_end = range_to_window_end
                    window_end_reached = True

                indices_within_range_window = np.where(
                    np.logical_and(
                        dem_to_sat_dists >= range_to_le_start,
                        dem_to_sat_dists <= range_to_le_end,
                    )
                )[0]

                if window_end_reached and window_start_reached:
                    if len(indices_within_range_window) == 0:
                        slope_ok[i] = False
                        break
                num_increments += 1
                if num_increments >= max_increments:
                    range_to_le_start = range_to_window_start
                    range_to_le_end = range_to_window_end

            if not slope_ok[i]:
                continue

            dem_to_sat_dists = np.array(dem_to_sat_dists)[indices_within_range_window]
            xdem = xdem[indices_within_range_window]
            ydem = ydem[indices_within_range_window]
            zdem = zdem[indices_within_range_window]

        # Find index of minimum range (ie heighest point) in remaining DEM points
        # and assign this as POCA
        index_of_closest = np.argmin(dem_to_sat_dists)
        if index_of_closest < 0 or index_of_closest > (len(xdem) - 1):
            slope_ok[i] = False
            continue

        poca_x[i] = xdem[index_of_closest]
        poca_y[i] = ydem[index_of_closest]
        poca_z[i] = zdem[index_of_closest]

        # Calculate the slope correction to height
        slope_correction[i] = (
            dem_to_sat_dists[index_of_closest] + poca_z[i] - altitudes[i]
        )

    # Transform all POCA x,y to lon,lat
    lon_poca_20_ku, lat_poca_20_ku = thisdem.xy_to_lonlat_transformer.transform(
        poca_x, poca_y
    )

    # Calculate height as altitude-(corrected range)+slope_correction
    height_20_ku = np.full_like(lat_20_ku, np.nan)

    num_measurements = len(lat_20_ku)
    for i in range(num_measurements):
        if np.isfinite(geo_corrected_tracker_range[i]):
            if slope_ok[i] and surface_type_20_ku[i] == 1:  # grounded ice type only
                height_20_ku[i] = (
                    altitudes[i]
                    - (geo_corrected_tracker_range[i] + retracker_correction[i])
                    + slope_correction[i]
                )
            else:
                height_20_ku[i] = np.nan
        else:
            height_20_ku[i] = np.nan
        # Set POCA lat,lon to nadir if no slope correction

        if (
            (not np.isfinite(lat_poca_20_ku[i]))
            or (not np.isfinite(lon_poca_20_ku[i]))
            or (not slope_ok[i])
        ):
            lat_poca_20_ku[i] = lat_20_ku[i]
            lon_poca_20_ku[i] = lon_20_ku[i]
            height_20_ku[i] = np.nan

    # ----------------------------------------------------------------
    # Doppler Slope Correction
    # ----------------------------------------------------------------

    if config["lrm_lepta_geolocation"]["include_slope_doppler_correction"]:
        idx = np.where(np.isfinite(height_20_ku))[0]
        if len(idx) > 0:
            ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
            lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
            this_transform = Transformer.from_proj(lla, ecef, always_xy=True)

            (  # pylint: disable=unpacking-non-sequence
                sat_x,
                sat_y,
                sat_z,
            ) = this_transform.transform(
                xx=lon_20_ku[idx],
                yy=lat_20_ku[idx],
                zz=altitudes[idx],
                radians=False,
            )
            (  # pylint: disable=unpacking-non-sequence
                ech_x,
                ech_y,
                ech_z,
            ) = this_transform.transform(
                xx=lon_poca_20_ku[idx],
                yy=lat_poca_20_ku[idx],
                zz=height_20_ku[idx],
                radians=False,
            )

            sdop = slope_doppler(
                sat_x,
                sat_y,
                sat_z,
                ech_x,
                ech_y,
                ech_z,
                l1b["sat_vel_vec_20_ku"][idx, :],
                config["instrument"]["chirp_slope"],
                config["instrument"]["wavelength"],
                config["geophysical"]["speed_light"],
            )

            height_20_ku[idx] += l1b["dop_cor_20_ku"][idx]
            height_20_ku[idx] -= sdop

    return (height_20_ku, lat_poca_20_ku, lon_poca_20_ku, slope_ok)