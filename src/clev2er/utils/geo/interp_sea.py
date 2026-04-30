"""Module for interpolating sea levels between sea ice leads.

interp_sea_regression(): Interpolates sea levels between leads using linear regression.

Author: Ben Palmer
Date: 18 Mar 2024
"""

import numpy as np
import numpy.typing as npt


def interp_sea_regression(
    lats_data: npt.NDArray,
    lons_data: npt.NDArray,
    sea_level_data: npt.NDArray,
    window_size: float,
) -> npt.NDArray:
    """Returns an array of interpolated sea levels using linear regression.

    Uses linear regression to find interpolated sea level values for arrays of lat, lon and sea
    level data. Uses a lead index to find leads within the window range of each point. To find a
    regressed value for a point, it must have at least one lead within window range on each side.

    Args:
        lats_data (npt.NDArray): Latitude data
        lons_data (npt.NDArray): Longitude data
        sea_level_data (npt.NDArray): Sea level data
        lead_index (npt.NDArray): Boolean array of leads
        window_size (float): Maximum distance that a point can be from the current point
        distance_projection (str, optional): Projection that is used when
            calculating distance. Defaults to "WGS84".

    Raises:
        ValueError: Raised if input arrays are different sizes on axis 0.
        ValueError: Raised if the lead_index input is not an array of bools.

    Returns:
        npt.NDArray: The array of regressed sea level values.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    if lats_data.shape[0] != lons_data.shape[0] and lats_data.shape[0] != sea_level_data.shape[0]:
        raise ValueError("Input arrays arrays do not have homogenous shape on axis 0.")

    out: npt.NDArray = np.zeros(lats_data.shape[0], dtype=int) * np.nan

    # NOTE: There are 2 versions of this function; the "Python-y" version and the  direct
    #       translation of Andy's code. I found that they produced the same results, but Andy's
    #       was much faster. Python-y version is O(n^2) while Andy's looks ~ O(n log n), could
    #       update Python-y version to match.

    # NOTE: Pythony version

    # for i in range(lats_data.shape[0]):
    #     if i in lead_index:
    #         out[i] = sea_level_data[i]

    #     point_distances: npt.NDArray = np.zeros((lats_data.shape[0])) * np.nan

    #     for j, (point_lat, point_lon) in enumerate(zip(lats_data, lons_data)):
    #         _, _, point_distances[j] = geod.inv(point_lon, point_lat, lons_data[i], lats_data[i])

    #     # finds leads within range window
    #     lead_and_in_window: npt.NDArray = lead_index & (point_distances <= window_size)

    #     # stops slicing index error
    #     slice_max: int = np.min([i + 1, point_distances.size - 1])

    #     points_bf: npt.NDArray = point_distances[:i][lead_and_in_window[:i]]
    #     points_af: npt.NDArray = point_distances[slice_max:][lead_and_in_window[slice_max:]]

    #     if ((points_bf.size) > 0) and ((points_af.size) > 0):
    #         # distances before current point should be negative during linear regression
    #         point_distances[:i] *= -1

    #         x: npt.NDArray = point_distances[lead_and_in_window].reshape(-1, 1)

    #         y: npt.NDArray = sea_level_data[lead_and_in_window]

    #         # y = y[np.invert(np.isnan(y))]

    #         lr = LinearRegression().fit(X=x, y=y)
    #         out[i] = lr.predict([[0]])[0]

    # NOTE: Andy's version

    for idx in range(lats_data.shape[0]):
        nlower = 0
        nupper = 0
        dist = 0.0

        x_arr = []
        y_arr = []

        for jdx in range(idx, lats_data.shape[0]):
            if not np.isnan(sea_level_data[jdx]):
                x_arr.append(dist)
                y_arr.append(sea_level_data[jdx])
                nupper += 1 if jdx > idx else 0
            # _, _, dist = geod.inv(lons_data[idx], lats_data[idx], lons_data[jdx], lats_data[jdx])
            dist = earth_dist(lats_data[idx], lons_data[idx], lats_data[jdx], lons_data[jdx])
            if dist > window_size:
                break

        for jdx in range(idx, -1, -1):
            if not np.isnan(sea_level_data[jdx]):
                x_arr.append(-dist)
                y_arr.append(sea_level_data[jdx])
                nlower += 1
            # _, _, dist = geod.inv(lons_data[idx], lats_data[idx], lons_data[jdx], lats_data[jdx])
            dist = earth_dist(lats_data[idx], lons_data[idx], lats_data[jdx], lons_data[jdx])
            if dist > window_size:
                break

        if nlower > 0 and nupper > 0:
            # lr = LinearRegression().fit(X=np.asarray(x_arr).reshape(-1, 1), y=np.asarray(y_arr))
            # out[idx] = lr.predict([[0]])[0]
            _, out[idx] = regress(np.asarray(x_arr), np.asarray(y_arr))
        else:
            out[idx] = np.nan

    return out


def earth_dist(dlat1, dlon1, dlat2, dlon2) -> float:
    """Finds the distance in meters between two points

    Author : Andy Ridout 13-OCT-2004
    Translate to Python: Ben Palmer 20-MAY-2025

    Args:
        dlat1 (float): Latitude in degrees
        dlon1 (float): Longitude in degrees
        dlat2 (float): Latitude in degrees
        dlon2 (float): Longitude in degrees

    Returns:
        float: distance in meters
    """
    lat1, lon1, lat2, lon2 = np.asarray([dlat1, dlon1, dlat2, dlon2]) * np.pi / 180.0

    tmp1 = np.sin(lat1) * np.sin(lat2)
    tmp2 = np.cos(lat1) * np.cos(lat2)
    tmp3 = np.cos(lon1 - lon2)
    return 6356.752 * np.arccos(tmp1 + (tmp2 * tmp3)) * 1000


def regress(x_arr: np.ndarray, y_arr: np.ndarray) -> tuple:
    """Fits a straight line through data by linear regression

    Author: Andy Ridout 11-MAR-2005
    Translate to Python: Ben Palmer 20-MAY-2025

    Args:
        x_arr (np.ndarray): Array of X values
        y_arr (np.ndarray): Array of Y values

    Returns:
        m: gradient of fitted line
        c: y-intercept of fitted line
    """
    xy = np.sum(x_arr * y_arr)
    x2 = np.sum(x_arr**2)
    x = np.sum(x_arr)
    y = np.sum(y_arr)

    c = ((xy * x) - (x2 * y)) / ((x * x) - (x_arr.size * x2))
    m = (xy - (x * c)) / x2
    return m, c
