"""Tool for plotting ASCII files output by the sea ice processor"""

import argparse
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pyproj


def plot_arctic_data(x, y, data, vmin=0, vmax=3.5):
    """Plots data from x, y, z arrays in arctic projection

    Args:
        x (ndarray): Array of x values
        y (ndarray): Array of y values
        data (ndarray): Array of data values
        vmin (int, optional): Minimum value on colourscale. Defaults to 0.
        vmax (float, optional): Maximum value on colourscale. Defaults to 3.5.

    Returns:
        Figure: matplotlib.pyplot.Figure instance
    """
    projection_new = ccrs.NorthPolarStereo(central_longitude=0)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(facecolor="white", projection=projection_new)

    ax.coastlines(resolution="110m", linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, color="black", alpha=0.4, linestyle="dashed", linewidth=0.5)
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    gl.ylabel_style = {
        "color": "black",
    }

    cs = ax.pcolormesh(
        x,
        y,
        data,
        vmax=vmax,
        vmin=vmin,
        cmap="viridis",
        transform=projection_new,
        shading="gouraud",
    )
    ax.add_feature(cartopy.feature.LAND, color="gainsboro", zorder=1)
    fig.colorbar(cs, ax=ax, label="Thickness(m)", extend="both")

    return fig


def read_ascii(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads in an ascii file, returns data as numpy arrays

    Args:
        filename (str): Path to the ascii file

    Returns:
        _type_: _description_
    """
    file_data = np.transpose(np.genfromtxt(filename))
    this_lats = file_data[2, :]
    this_lons = file_data[3, :]
    this_thickness = file_data[4, :]
    this_number_in = file_data[5, :]

    for name, arr in zip(
        ["Lats", "Lons", "Thickness", "Number in"],
        [this_lats, this_lons, this_thickness, this_number_in],
    ):
        print(
            f"{name}: min={np.nanmin(arr):0.3f} max={np.nanmax(arr):0.3f} "
            f"mean={np.nanmean(arr):0.3f} "
            f"nans={np.sum(np.isnan(arr))}"
        )

    return this_lats, this_lons, this_thickness, this_number_in


if __name__ == "__main__":
    # pylint:disable=unpacking-non-sequence
    # pylint:disable=invalid-name
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--input_file", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path)

    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Cannot find input file {args.input_file}")

    if not args.output_dir.exists():
        print("Output file parent directory does not exist, creating...")
        args.output_dir.mkdir(parents=True)

    lats, lons, thickness, number_in = read_ascii(args.input_file)

    crs_old = pyproj.CRS.from_epsg(4326)
    crs_new = pyproj.CRS.from_epsg(3413)
    print(crs_old.axis_info)
    print(crs_new.axis_info)
    latlon_to_xy = pyproj.Transformer.from_crs(crs_old, crs_new)

    y_values, x_values = latlon_to_xy.transform(lats, lons)

    print(
        f"X: min={np.nanmin(x_values):0.3f} max={np.nanmax(x_values):0.3f} "
        f"mean={np.nanmean(x_values):0.3f} nans={np.sum(np.isnan(x_values))}"
    )
    print(
        f"Y: min={np.nanmin(y_values):0.3f} max={np.nanmax(y_values):0.3f} "
        f"mean={np.nanmean(y_values):0.3f} nans={np.sum(np.isnan(y_values))}"
    )

    x_values = x_values.reshape((500, 720))
    y_values = y_values.reshape((500, 720))
    thickness = thickness.reshape((500, 720))
    number_in = number_in.reshape((500, 720))

    thickness_plot: matplotlib.figure.Figure = plot_arctic_data(
        x_values, y_values, thickness, None, None
    )
    nin_plot: matplotlib.figure.Figure = plot_arctic_data(x_values, y_values, number_in, vmax=5)

    thickness_plot.savefig(args.output_dir / "thickness_plot.png")
    nin_plot.savefig(args.output_dir / "number_in.png")
