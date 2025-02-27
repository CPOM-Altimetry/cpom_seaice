"""Module for calculating grid cell positions using latitude and longitude values
    and vice versa.
get_cell_indexes_from_lat_lon: Returns cell position from input lat/lon pairs
get_lat_lon_from_cell_indexes: Returns the lat/lon position of input cell location pairs
    """
import numpy as np
import numpy.typing as npt


def get_cell_indexes_from_lat_lon(
    lats: npt.ArrayLike, lons: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates cell indexes for lat lon pairs

    Args:
        lats (np.ndarray | float): Latitude values
        lons (np.ndarray | float): Longitude values

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple of (ilats, ilons)
    """
    if not isinstance(lats, np.ndarray):
        lats = np.asarray(lats)
    if not isinstance(lons, np.ndarray):
        lons = np.asarray(lons)

    ilats = ((lats - 40) / 0.1).astype(int)
    ilons = (((lons + 180) % 360) / 0.5).astype(int)
    # line above is weird. Andy uses coordinates -180..180, we use 0..360
    # but andy's ilon formula only works right now with -180..180
    # to convert from former to latter, use: (lon + 180) % 360 - 180
    # andy's formula for lon to ilon is: (lon + 180) / 0.5
    # substituting the coordinate conversion into the ilon conversion, we get
    # ilon = ((lon + 180) % 360 - 180 + 180) / 0.5
    # can simplify as
    # ilon = ((lon + 180) % 360) / 0.5
    # you can probably simplify this by changing the ilon formula to work with
    # 0..360 values, but thats a problem for another day

    return ilats, ilons


def get_lat_lon_from_cell_indexes(
    ilats: npt.ArrayLike, ilons: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates lat lon values from cell index pairs

    Args:
        ilats (np.array | int): ilat values
        ilons (np.array | int): ilon values

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple of (lats, lons)
    """
    if not isinstance(ilats, np.ndarray):
        ilats = np.asarray(ilats)
    if not isinstance(ilons, np.ndarray):
        ilons = np.asarray(ilons)
    lats = 40 + (ilats * 0.1)
    lons = -180 + (ilons * 0.5)

    return lats, lons
