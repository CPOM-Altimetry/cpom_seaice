"""Module for interpolating sea levels between sea ice leads.

interp_sea_regression(): Interpolates sea levels between leads using linear regression.

Author: Ben Palmer
Date: 18 Mar 2024
"""

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import BallTree


def interp_sea_regression_tree(
    lats_data,
    lons_data,
    sea_level_data,
    window_size: float,
    earth_radius: float,
):
    """Returns an array of interpolated sea levels using linear regression.

    Uses linear regression to find interpolated sea level values for arrays of lat, lon and sea
    level data. Uses a lead index to find leads within the window range of each point. To find a
    regressed value for a point, it must have at least one lead within window range on each side.
    This version uses a sklean.BallTree to find the points, which is significantly faster than
    the previous version.

    Args:
        lats_data (npt.NDArray): Latitude data
        lons_data (npt.NDArray): Longitude data
        sea_level_data (npt.NDArray): Sea level data
        window_size (float): Maximum distance that a point can be from the current point

    Raises:
        ValueError: Raised if input arrays are different sizes on axis 0.
        ValueError: Raised if the lead_index input is not an array of bools.

    Returns:
        npt.NDArray: The array of regressed sea level values.
    """
    # pylint: disable=too-many-locals
    if lats_data.shape[0] != lons_data.shape[0] and lats_data.shape[0] != sea_level_data.shape[0]:
        raise ValueError("Input arrays arrays do not have homogenous shape on axis 0.")

    out = np.zeros(lats_data.shape[0], dtype=int) * np.nan

    data_points = (np.transpose([lats_data, lons_data]) * np.pi) / 180
    valid_data = np.isfinite(sea_level_data)
    where_points = np.where(valid_data)[0]
    tree_points = data_points[valid_data]
    interp_tree = BallTree(tree_points, metric="haversine")

    nearest_points, distances = interp_tree.query_radius(
        data_points, r=window_size / earth_radius, return_distance=True, sort_results=True
    )

    distances *= earth_radius

    for i, (this_nearest_points, this_dist) in enumerate(zip(nearest_points, distances)):
        if not this_nearest_points.size > 0:
            continue
        actual_indices = where_points[this_nearest_points]

        before_i = actual_indices < i
        after_i = actual_indices > i
        if not before_i.any() or not after_i.any():
            continue

        interp_values = np.concatenate(
            [sea_level_data[actual_indices[before_i]], sea_level_data[actual_indices[after_i]]]
        )
        interp_distances = np.concatenate([this_dist[before_i] * -1, this_dist[after_i]])

        _, out[i] = regress(interp_distances, interp_values)
    return out


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
        window_size (float): Maximum distance that a point can be from the current point

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

    out: npt.NDArray = np.zeros(lats_data.shape[0], dtype=float) * np.nan

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

        for jdx in range(idx + 1, lats_data.shape[0]):
            dist = earth_dist(lats_data[idx], lons_data[idx], lats_data[jdx], lons_data[jdx])
            if dist > window_size:
                break
            if not np.isnan(sea_level_data[jdx]):
                x_arr.append(dist)
                y_arr.append(sea_level_data[jdx])
                nupper += 1 if jdx > idx else 0

        for jdx in range(idx - 1, -1, -1):
            dist = earth_dist(lats_data[idx], lons_data[idx], lats_data[jdx], lons_data[jdx])
            if dist > window_size:
                break
            if not np.isnan(sea_level_data[jdx]):
                x_arr.append(-dist)
                y_arr.append(sea_level_data[jdx])
                nlower += 1

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


def earth_dist_points(points1, points2) -> float:
    """Finds the distance in km between two points

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

    # lat lon points might be almost exactly the same but differ by incredibly
    # small amounts, so check if they're equal to 4 decimal places
    if np.abs(np.sum(points1 - points2)) < 0.0001:
        return 0

    dist = 6356.752 * np.arccos(
        np.sin(points1[0]) * np.sin(points2[0])
        + (np.cos(points1[0]) * np.cos(points2[0]) * np.cos(points1[1] - points2[1]))
    )
    return dist


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
