"""Module for calculating grid cell positions using latitude and longitude values
    and vice versa.
get_cell_indexes_from_lat_lon: Returns cell position from input lat/lon pairs
get_lat_lon_from_cell_indexes: Returns the lat/lon position of input cell location pairs
    """
import numpy as np


@np.vectorize
def get_cell_indexes_from_lat_lon(lat: float, lon: float) -> tuple[int, int]:
    """Calculates cell indexes for lat lon pairs

    Args:
        lats (float): Latitude values
        lons (float): Longitude values

    Returns:
        tuple[int, int]: tuple of (ilats, ilons)
    """

    ilat = np.floor((lat - 40) / 0.1).astype(int)
    ilon = np.floor((lon + 180) / 0.5).astype(int)
    return ilat, ilon


@np.vectorize
def get_lat_lon_from_cell_indexes(ilat: int, ilon: int) -> tuple[float, float]:
    """Calculates lat lon values from cell index pairs

    Args:
        ilats (int): ilat values
        ilons (int): ilon values

    Returns:
        tuple[float, float]: tuple of (lats, lons)
    """
    lat = 40 + (ilat * 0.1)
    lon = -180 + (ilon * 0.5)

    return lat, lon
