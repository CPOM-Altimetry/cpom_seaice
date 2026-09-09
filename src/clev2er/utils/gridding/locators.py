"""Module for calculating grid cell positions using latitude and longitude values
    and vice versa.
get_cell_indexes_from_lat_lon: Returns cell position from input lat/lon pairs
get_lat_lon_from_cell_indexes: Returns the lat/lon position of input cell location pairs
    """
import numpy as np

RADIUS = 6378.273  # Earth's radius
PANGLE = np.radians(70.0)  # Latitude of projection plane
PI = 3.1415926535
E = 0.081816153  # Eccentricity of ellipsoid


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
    lat = 40 + ilat * 0.1
    lon = -180 + ilon * 0.5

    return lat, lon


# This are used by the original C script to project Lat/Lon to x/y
# They're used to ensure that the functionality is carried forward to match the original


@np.vectorize
def proj_ll2xy_s(lat: float, lon) -> tuple[float, float]:
    """Projects lat/lon to x/y for the southern hemisphere (EPSG:3413)

    Args:
        lat (float): Latitude in degrees N (-90 -> 90)
        lon (float): Longitude in degrees E (-180 -> 180)

    Raises:
        RuntimeError: Raised if the longitude is outside of the -180->180 bounds

    Returns:
        tuple(float,float): Projected x/y coordinates
    """
    lat *= -1.0

    if lon > 180.0 or lon < -180.0:
        raise RuntimeError("project_ll2xy_s-> Longitude outside range -180 to +180\n")

    if lon > 0:
        lon = 180.0 - lon
    else:
        lon = -180.0 - lon

    x, y = proj_ll2xy(lat, lon)

    return x, y


@np.vectorize
def proj_ll2xy(dlat: float, dlon: float) -> tuple[float, float]:
    """Projects lat/lon to x/y for the northern hemisphere (EPSG:3031)

    Args:
        dlat (float): Latitude in degrees N (-90 -> 90)
        dlon (float): Longitude in degrees E (-180 -> 180)

    Returns:
        tuple[float, float]: Projected x/y coordinates
    """
    lat = np.radians(dlat)
    lon = np.radians(dlon)

    rho = (RADIUS * _gammafunction(PANGLE) * _tfunction(lat)) / _tfunction(PANGLE)

    x: float = rho * np.sin(lon)
    y: float = rho * np.cos(lon) * -1.0

    return x, y


@np.vectorize
def _tfunction(w: float) -> float:
    """T function used in conformal map projections. Internal use for ll2ps functions

    Args:
        w (float): Latitude in radians

    Returns:
        float: Value of the t function at latitude w
    """
    topline = np.tan((PI / 4.0) - (w / 2.0))
    botline = ((1.0 - E * np.sin(w)) / (1.0 + E * np.sin(w))) ** (E / 2.0)

    outval = topline / botline

    return outval


@np.vectorize
def _gammafunction(phi: float) -> float:
    """Gamma function used in conformal map projections. Internal use for ll2ps functions

    Args:
        phi (float): Latitude in radians

    Returns:
        float: Value of gamma given latitude phi
    """
    topline = np.cos(phi)
    botline = np.sqrt(1.0 - E * E * np.sin(phi) * np.sin(phi))

    outval = topline / botline

    return outval
