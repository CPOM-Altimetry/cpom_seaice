"""Pytest file for clev2er.utils.gridding.locators"""

from clev2er.utils.gridding.locators import (
    get_cell_indexes_from_lat_lon,
    get_lat_lon_from_cell_indexes,
)


def test_ll_conversion():
    """Test for lat/lon conversion functions

    Plan:
    - Get lat/lon for known cell index
    - Get cell index from converted lat/lon
    - Check if returned cell indices is the original indices
    - Test if extents match expected grid cell numbers
    """
    lat, lon = get_lat_lon_from_cell_indexes(0, 0)

    cell_x, cell_y = get_cell_indexes_from_lat_lon(lat, lon)

    assert (
        cell_x == 0 and cell_y == 0
    ), f"Error in conversion, expected (0,0) but got {(cell_x, cell_y)}"

    cell_x, cell_y = get_cell_indexes_from_lat_lon(90, 0)
    print(cell_x, cell_y)
    assert cell_x == 499, f"Error in conversion, expected 499 but got {cell_x}"

    cell_x, cell_y = get_cell_indexes_from_lat_lon(90, 360)
    print(cell_x, cell_y)
    assert cell_y == 719, f"Error in conversion, expected 719 but got {cell_y}"
