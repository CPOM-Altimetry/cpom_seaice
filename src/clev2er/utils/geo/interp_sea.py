""" Module for interpolating sea levels between sea ice leads.

    interp_sea_regression(): Interpolates sea levels between leads using linear regression.
    
    Author: Ben Palmer
    Date: 18 Mar 2024
"""

import numpy as np
import numpy.typing as npt
import pyproj as proj
from sklearn.linear_model import LinearRegression


def interp_sea_regression(
    lats_data: npt.NDArray,
    lons_data: npt.NDArray,
    sea_level_data: npt.NDArray,
    lead_index: npt.NDArray,
    window_size: float,
    distance_projection: str = "WGS84",
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
    if (
        lats_data.shape[0] != lons_data.shape[0]
        and lats_data.shape[0] != sea_level_data.shape[0]
        and lats_data.shape[0] != lead_index.shape[0]
    ):
        raise ValueError("Input arrays arrays do not have homogenous shape on axis 0.")

    if "bool" not in str(lead_index.dtype).lower():
        raise ValueError("Lead_index array must contain bool values")

    geod: proj.Geod = proj.Geod(ellps=distance_projection)

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
        dist = 0

        x_arr = []
        y_arr = []

        for jdx in range(idx, lats_data.shape[0]):
            if lead_index[jdx]:
                x_arr.append(dist)
                y_arr.append(sea_level_data[jdx])
                nupper += 1 if jdx > idx else 0
            _, _, dist = geod.inv(lons_data[idx], lats_data[idx], lons_data[jdx], lats_data[jdx])
            if dist > window_size:
                break

        for jdx in range(idx, -1, -1):
            if lead_index[jdx]:
                x_arr.append(-dist)
                y_arr.append(sea_level_data[jdx])
                nlower += 1
            _, _, dist = geod.inv(lons_data[idx], lats_data[idx], lons_data[jdx], lats_data[jdx])
            if dist > window_size:
                break

        if nlower > 0 and nupper > 0:
            lr = LinearRegression().fit(X=np.asarray(x_arr).reshape(-1, 1), y=np.asarray(y_arr))
            out[idx] = lr.predict([[0]])[0]
        else:
            out[idx] = np.nan

    return out
