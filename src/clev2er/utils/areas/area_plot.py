"""clev2er.utils.areas.area_plot.py
class to plot areas defined in clev2er.utils.areas.definitions

To do reminder:

simple square plots
doc in __init__.py
grid support
vostok area
arctic area
"""

import logging
import os
from dataclasses import dataclass

import cartopy.crs as ccrs  # type: ignore
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import shapefile as shp  # type: ignore
from cartopy.mpl.geoaxes import GeoAxesSubplot  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ma  # masked arrays

from clev2er.utils.areas.areas import Area
from clev2er.utils.backgrounds.backgrounds import (  # background images functions for polar plots
    Background,
)

# pylint:disable=unpacking-non-sequence
# pylint:disable=too-many-lines
# pylint:disable=too-many-branches
# pylint:disable=too-many-statements
# pylint:disable=too-many-arguments
# pylint:disable=too-many-locals

log = logging.getLogger(__name__)


def get_unique_colors(n: int, cmap_name_override: str | None = None):
    """get a list of n unique colors for plotting flag data (when no colors are
       provided, as sampled from the tab20 or tab10 colormap

    Args:
        n (int): number of colors required (<= 20 will provide unique colors
        otherwise some repetition)
        cmap_name_override (str): override colormap name to use, Typical alternatives are
                         "tab10", "tab20b", "tab20c", and "Set1", "Set2","Set3"

    Returns:
        List[Tuple[float,float,float,float]]: list of color RGBA
    """
    if n <= 10:
        cmap_name = "tab10"
    else:
        cmap_name = "tab20"
    if cmap_name_override is not None:
        cmap_name = cmap_name_override
    cmap = plt.cm.get_cmap(cmap_name, n)  # Replace 'tab20' with any suitable colormap
    colors = [cmap(i) for i in range(cmap.N)]
    return colors


def calculate_mad(values: np.ndarray):
    """Calculate the MAD (Mean Absolute Deviation) of values
    Args:
        values (np.ndarray): values for which MAD is to be calculated
    Returns:
        mad (float) : MAD value
    """
    values_np = np.array(values)
    mean_value = np.mean(values_np)
    absolute_deviations = np.abs(values_np - mean_value)
    mad = np.mean(absolute_deviations)
    return mad


@dataclass
class Annotation:
    """
    Data class for a single annotation in a polar plot.

    Attributes:
        xpos (float): The x-coordinate position of the annotation text in axis coordinates (0-1),
                      representing the percentage of the axis width.
        ypos (float): The y-coordinate position of the annotation text in axis coordinates (0-1),
                      representing the percentage of the axis height.
        text (str): The content of the annotation text.
        bbox (dict | None): An optional bounding box around the text. If provided, it should be a
                            dictionary specifying the style of the box. Default is None, meaning no
                            box. Example format:
                                {
                                    'boxstyle': 'round',  # Style of the box (e.g.,'round','square')
                                    'facecolor': 'aliceblue',  # Background color of the box
                                    'alpha': 1.0,  # Transparency of the box (0-1)
                                    'edgecolor': 'lightgrey',  # Color of the box edge
                                }
                            See matplotlib Boxstyle documentation for more styles.
        fontsize (int): The font size of the text. Default is 12.
        color (str): The color of the text. Default is "k" (black).
        fontweight (str): The weight (thickness) of the font. Default is "normal".
    """

    xpos: float
    ypos: float
    text: str
    bbox: dict | None = None
    fontsize: int = 12
    color: str = "k"
    fontweight: str = "normal"


class Polarplot:
    """class to create map plots of polar areas"""

    def __init__(self, area: str, area_overrides: dict | None = None):
        """class inititialization

        Args:
            area (str): area name as per clev2er.utils.areas.definitions
            area_overrides (dict|None): dictionary to override area dict definitions
        """
        self.area = area

        self.thisarea = Area(area, area_overrides)

        if self.thisarea.mask:
            self.thismask = self.thisarea.mask

    def plot_points(
        self,
        *data_sets,
        use_default_annotation: bool = True,
        annotation_list: list[Annotation] | None = None,
        logo_image=None,
        logo_position: tuple[float, float, float, float] | None = None,
        output_dir: str = "",
        output_file: str = "",
        dpi: int = 85,
        transparent_background: bool = False,
        map_only: bool = False,
    ):
        """function to plot one or more (lat,lon,val) datasets on polar maps


        Args:
            *data_sets (dict, optional) : data set dictionaries (you can have more than
                    one dataset plotted by providing dataset1_dict,dataset2_dict2,...)
                    Each dictionary contains the (lats,lons,vals) for a data set, and
                    the optional tunable plot parameters for that data set only.
                    Each data_set should consist of
                    {
                        # Required key/vals:
                        "lats": np.array([]),
                        "lons": np.array([]),
                        "vals": np.array([]),
                        #-----------------------------------------------
                        # Optional, otherwise default values are used.
                        #-----------------------------------------------
                        "units": '', # units of vals
                        "name": "unnamed", # str name of vals. Used to label plot.
                                                 # 'unnamed' is used if not provided.
                        "apply_area_mask_to_data": True, # bool, whether to apply the default area's
                                                        # data Mask to this data set
                        # --- flagging bad data for this data set, plotted in mini-map
                        "fill_value": 9999, # fill value in vals to be ignored or None
                        "valid_range": [min,max],# [min,max] or None. allowed vals range. flagged as
                                                 # bad outside this range.
                                                 # default is actual min,max of vals
                        "minimap_val_scalefactor": 1.,# (float) scale the default
                                                      # plot marker for bad data
                        # -- flag type data settings --------------------------------
                        "flag_values": [], # list of flag values. If used vals treated as flag data
                        "flag_names": [], # list of flag names
                        "flag_colors": [] # list of flag colors or colormap name to sample
                        # --- color map, color bar
                        "cmap_name": "RdYlBu_r", # colormap name to use for this dataset
                        "cmap_over_color": "#A85754" or None
                        "cmap_under_color": "#3E4371"  or None
                        "cmap_extend": "both" # 'neither','min', 'max','both'
                        "min_plot_range": None, # set the minimum range for the colorbar.
                                           # if not set min(vals) will be used
                        "max_plot_range": None, # set the maximum range for the colorbar
                        # --- point size, alpha
                        "plot_size_scale_factor": 1., # (float) scale the default plot marker
                        "plot_alpha": 1.0, # transparency of this dataset plot (0..1)
                    }
            use_default_annotation (bool): if True display default dataset annotation else do not
            annotation_list (list[Annotation]|None, optional): list of Annotation objects to display
            logo_image (,optional): logo image to insert in plot as returned by
                                    plt.imread('someimagefile.png')
            logo_position (list,optional) : logo position as an axis rect list:
                                        [left, bottom, width, height] , each are 0..1

            output_dir (str,optional):  output directory to save plots instead of displaying them.
                              if output_file not specified, plot saved with name of 1st data_set and
                              area: <output_dir>/param_<data_set['name']>_<self.area>.png
            output_file (str,optional): optionally override default output plot .png filename.
                              if output_dir specified, saved as <output_dir>/output_file
                              if no output_dir than output_file should contain the full path
            dpi (int,optional): dpi to save image, default=85
            transparent_background (bool, optional): set to have transparent background when saved
                                 as png
            map_only (bool): plot just the map plot (+ colorbar). No histograms or other elements.

        Raises:
            ValueError: if data_set parameters (lat,lon,vals) do not have equal length
        """

        # -----------------------------------------------------------------------------------------
        # Default plot configuration
        # -----------------------------------------------------------------------------------------

        # Default plot parameters. If these are set to None, then the equivalent settings
        # in self.thisarea are used instead if set
        plot_params = {
            "fig_width": 12,
            "fig_height": 10,
            "draw_axis_frame": False,  # (bool|None)
            #  alpha value transparency (alpha value) of background. Float between 0 and
            #   1, or list of floats for multiple backgrounds.
        }
        if map_only:
            plot_params["fig_width"] = 10
            self.thisarea.include_flag_legend = False
            self.thisarea.include_flag_percents = False
            self.thisarea.show_bad_data_map = False
            self.thisarea.show_minimap = False
            self.thisarea.show_histograms = False
            self.thisarea.show_latitude_scatter = False
            self.thisarea.axes = self.thisarea.simple_axes
            self.thisarea.area_long_name_position = self.thisarea.area_long_name_position_simple
            self.thisarea.mask_long_name_position = self.thisarea.mask_long_name_position_simple
            self.thisarea.horizontal_colorbar_axes = self.thisarea.horizontal_colorbar_axes_simple
            self.thisarea.vertical_colorbar_axes = self.thisarea.vertical_colorbar_axes_simple
            self.thisarea.stats_position_y_offset = self.thisarea.stats_position_y_offset_simple
            self.thisarea.stats_position_x_offset = self.thisarea.stats_position_x_offset_simple
            self.thisarea.varname_annotation_position_xy = (
                self.thisarea.varname_annotation_position_xy_simple
            )
        # ------------------------------------------------------------------------------------------
        # Setup plot page
        # ------------------------------------------------------------------------------------------

        fig = plt.figure(figsize=(plot_params["fig_width"], plot_params["fig_height"]))
        # width, height in inches

        # ------------------------------------------------------------------------------------------
        # Load annotations
        #    - 2 types of annotation
        #        - default annotation  (can be disabled)
        #        - custom annotation
        # ------------------------------------------------------------------------------------------

        annot_ax = fig.add_axes(
            (0.0, 0.0, 1.0, 1.0), zorder=10
        )  # left, bottom, width, height (fractions of axes)
        annot_ax.get_xaxis().set_visible(False)
        annot_ax.get_yaxis().set_visible(False)
        annot_ax.axis("off")

        # Draw all the annotations

        final_annotation_list = []
        if annotation_list is not None and len(annotation_list) > 0:
            final_annotation_list.extend(annotation_list)
        if use_default_annotation:
            if len(data_sets) > 0:
                final_annotation_list.append(
                    Annotation(
                        self.thisarea.varname_annotation_position_xy[0],
                        self.thisarea.varname_annotation_position_xy[1],
                        data_sets[0].get("name", "unnamed"),
                        {
                            "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                            "facecolor": "aliceblue",  # Background color of the box
                            "alpha": 1.0,  # Transparency of the box (0-1)
                            "edgecolor": "lightgrey",  # Color of the box edge
                        },
                        18,
                    )
                )
                final_annotation_list.append(
                    Annotation(
                        self.thisarea.varname_annotation_position_xy[0] - 0.01,
                        self.thisarea.varname_annotation_position_xy[1] + 0.04,
                        "Variable plotted:",
                        fontsize=9,
                    )
                )
                if self.thisarea.area_long_name_position is not None:
                    final_annotation_list.append(
                        Annotation(
                            self.thisarea.area_long_name_position[0],
                            self.thisarea.area_long_name_position[1],
                            self.thisarea.long_name,
                            fontsize=self.thisarea.area_long_name_fontsize,
                        )
                    )

                if self.thisarea.mask_long_name_position is not None:
                    if self.thisarea.maskname is not None:
                        if bool(data_sets[0].get("apply_area_mask_to_data", False)):
                            final_annotation_list.append(
                                Annotation(
                                    self.thisarea.mask_long_name_position[0],
                                    self.thisarea.mask_long_name_position[1],
                                    f"Data mask: {self.thisarea.maskname}",
                                    fontsize=self.thisarea.mask_long_name_fontsize,
                                )
                            )
                        else:
                            final_annotation_list.append(
                                Annotation(
                                    self.thisarea.mask_long_name_position[0],
                                    self.thisarea.mask_long_name_position[1],
                                    "Data mask applied: None",
                                    fontsize=self.thisarea.mask_long_name_fontsize,
                                )
                            )

        if final_annotation_list is not None:
            for annot in final_annotation_list:
                annot_ax.text(
                    annot.xpos,
                    annot.ypos,
                    annot.text,
                    fontsize=annot.fontsize,
                    fontweight=annot.fontweight,
                    color=annot.color,
                    transform=annot_ax.transAxes,
                    bbox=annot.bbox,
                )

        # -----------------------------------------------------------------------------
        # Logo Images
        # -----------------------------------------------------------------------------

        if logo_image is not None:
            if logo_position is None:
                logo_position = (0.0, 0.0, 1.0, 1.0)
            newax = fig.add_axes(logo_position, frameon=False)
            newax.imshow(logo_image, zorder=1)
            newax.axis("off")

        # ------------------------------------------------------------------------------------------
        # Setup Axis, Projection and Extent for Area  (north or south polar stereo)
        #    either by areas:
        #       lower left corner lat,lon, width, height
        #       centre lat,lon, width, height
        #       bounding latitude for circular areas around pole
        # ------------------------------------------------------------------------------------------

        ax, dataprj, circle = self.setup_projection_and_extent(
            self.thisarea.axes, draw_axis_frame=plot_params["draw_axis_frame"]
        )

        # ------------------------------------------------------------------------------------------
        # Set map backgrounds
        # ------------------------------------------------------------------------------------------

        ax.set_facecolor(self.thisarea.background_color)

        # ------------------------------------------------------------------------------------------
        # Add a Background image
        #    from background parameter which can be: None, single_str, [str1, str2]
        #       if None: use self.thisarea.background_image
        #       if single_str, use this for background string
        #       if [str1, str2], apply backgrounds str1, str2 in order given
        #
        #   background_alpha=None, # transparency (alpha value) of background. Float between 0 and
        #   1, or list of floats for multiple backgrounds.
        #   if None, then use default value for area: self.thisarea.background_image_alpha
        # ------------------------------------------------------------------------------------------

        # use default background if no background provided

        background = self.thisarea.background_image

        # form a list of backgrounds (even if single background)
        if isinstance(background, list):
            backgrounds = background
        else:
            backgrounds = [background]

        # form a list of background_alpha names to apply
        background_alpha = self.thisarea.background_image_alpha

        if isinstance(background_alpha, list):
            background_alphas = np.array(background_alpha)
        else:
            background_alphas = np.array([background_alpha])

        # form a list of background_resolution names to apply
        background_resolution = self.thisarea.background_image_resolution

        if isinstance(background_resolution, list):
            background_resolutions = background_resolution
        else:
            background_resolutions = [background_resolution]

        if len(background_alphas) != len(backgrounds):
            log.warning("Number of background_alphas does not match number of backgrounds.")
            background_alphas = np.full(len(backgrounds), background_alphas[0])

        if len(background_resolutions) != len(backgrounds):
            log.warning("Number of background_resolutions does not match number of backgrounds.")
            background_resolutions = background_resolutions * len(backgrounds)

        # display each background in turn
        hillshade_params = self.thisarea.hillshade_params
        for i, thisbackground in enumerate(backgrounds):
            if isinstance(hillshade_params, dict):
                bg_hillshade_params = hillshade_params.copy()
                bg_hillshade_params.update({"alpha": background_alphas[i]})
            else:
                bg_hillshade_params = {"alpha": background_alphas[i]}
            Background(thisbackground, self.thisarea).load(
                ax,
                dataprj,
                resolution=background_resolutions[i],
                alpha=background_alphas[i],
                hillshade_params=bg_hillshade_params,
            )

        # ------------------------------------------------------------------------------------------
        #   Overlay mask polygon
        # ------------------------------------------------------------------------------------------

        self.draw_area_polygon_mask(
            ax, self.thisarea.show_polygon_mask, self.thisarea.polygon_mask_color, dataprj
        )

        #  draw mini-map of bad values
        self.draw_minimap()

        # ------------------------------------------------------------------------------------------
        # Load data sets
        # ------------------------------------------------------------------------------------------
        ds_name_0 = "unnamed"
        num_data_sets = len(data_sets)
        if num_data_sets > 0:
            log.info("Loading %d data sets", num_data_sets)

            for ds_num, data_set in enumerate(data_sets):
                print(f"loading data set {ds_num}: {data_set.get('name','unnamed')}")

                is_flag_data = len(data_set.get("flag_values", [])) > 0

                lats = data_set.get("lats", np.array([]))
                lons = data_set.get("lons", np.array([]))
                vals = data_set.get("vals", np.array([]))

                n_vals = len(vals)

                if n_vals != len(lats):
                    raise ValueError(
                        f"length of vals array must equal lats array in data set {ds_num}"
                    )
                if n_vals != len(lons):
                    raise ValueError(
                        f"length of vals array must equal lons array in data set {ds_num}"
                    )

                # convert to ndarray if a list
                if not isinstance(lats, np.ndarray):
                    lats = np.asarray(lats)
                if not isinstance(lons, np.ndarray):
                    lons = np.asarray(lons)
                if not isinstance(vals, np.ndarray):
                    vals = np.asarray(vals)

                # Check if data is not 1-d. If n-d > 1, flatten to 1-d
                if len(np.shape(lats)) > 1:
                    lats = lats.flatten()
                if len(np.shape(lons)) > 1:
                    lons = lons.flatten()
                if len(np.shape(vals)) > 1:
                    vals = vals.flatten()

                # ------------------------------------------------------------------------------
                # check lats,lons for valid values before plotting
                # ------------------------------------------------------------------------------

                # Convert None to np.nan and ensure the array is of float type
                lats = np.array(lats, dtype=float)
                lons = np.array(lons, dtype=float)

                # Test for masked arrays
                if ma.is_masked(lats):
                    lats[lats.mask.nonzero()] = np.nan
                if ma.is_masked(lons):
                    lons[lons.mask.nonzero()] = np.nan

                # Step 1: Filter for valid values
                # Assuming latitude values must be between -90 and 90, and longitude
                # between -180 and 180 or 0 to 360
                valid_lat = (lats >= -90) & (lats <= 90)
                valid_long = (lons >= -180) & (lons <= 360)

                # Handling NaNs or None for both lats and lons
                valid_lat = valid_lat & ~np.isnan(lats)
                valid_long = valid_long & ~np.isnan(lons)

                # Step 2: Normalize Longitudes
                # Convert lons from -180 to 180 to 0 to 360
                lons = np.where(lons < 0, lons + 360, lons)

                # Step 3: Identify common indices
                valid_indices = np.where(valid_lat & valid_long)[0]

                if valid_indices.size == 0:
                    log.error(
                        "No valid latitude and longitude values in dataset %s",
                        data_set.get("name", f"unnamed_{ds_num}"),
                    )
                    continue

                lats = lats[valid_indices]
                lons = lons[valid_indices]
                vals = vals[valid_indices]

                log.info(
                    "%d valid lat/lon values found for dataset %s",
                    lats.size,
                    data_set.get("name", f"unnamed_{ds_num}"),
                )

                # ------------------------------------------------------------------------------
                # Area Mask data sets
                # ------------------------------------------------------------------------------

                apply_area_mask = data_set.get(
                    "apply_area_mask_to_data", self.thisarea.apply_area_mask_to_data
                )

                if apply_area_mask:
                    log.info("Masking data with area's data mask..")

                    lats, lons, inside_area, n_inside = self.thisarea.inside_area(lats, lons)
                    if n_inside > 0:
                        vals = vals[inside_area]
                    else:
                        log.error("No data inside mask for data set %d", ds_num)
                        continue
                    log.info("Number of values inside mask %d of %d", n_inside, n_vals)
                else:
                    log.info("No data mask applied")

                # ------------------------------------------------------------------------------
                # Check vals for Nan and FillValue before plotting
                # ------------------------------------------------------------------------------

                # convert None to Nan
                try:
                    vals = np.array(vals, dtype=float)
                except ValueError:
                    log.error("invalid value type in dataset found. Must be int or float")
                    continue

                # find Nan values in data ------------------------------------------------------
                nan_vals_bool = np.isnan(vals)
                percent_nan = np.mean(nan_vals_bool) * 100.0
                nan_indices = np.where(nan_vals_bool)[0]
                if nan_indices.size > 0:
                    nan_lats = lats[nan_indices]
                    nan_lons = lons[nan_indices]
                else:
                    nan_lats = np.array([])
                    nan_lons = np.array([])

                log.info("percent Nan %.2f", percent_nan)

                # find out of range values in data -------------------------------------------------

                if is_flag_data:
                    outside_vals_bool = (vals < np.min(data_set.get("flag_values"))) | (
                        vals > np.max(data_set.get("flag_values"))
                    )
                    percent_outside = np.mean(outside_vals_bool) * 100.0
                else:
                    if (
                        data_set.get("valid_range") is not None
                        and len(data_set.get("valid_range")) != 2
                    ):
                        log.error("valid_range plot parameter must be of type [min,max]")
                    if (
                        data_set.get("valid_range") is not None
                        and len(data_set.get("valid_range")) == 2
                    ):
                        outside_vals_bool = (vals < data_set.get("valid_range")[0]) | (
                            vals > data_set.get("valid_range")[1]
                        )
                        percent_outside = np.mean(outside_vals_bool) * 100.0
                    else:
                        percent_outside = 0.0
                        outside_vals_bool = np.full_like(vals, False, bool)

                outside_indices = np.where(outside_vals_bool)[0]
                if outside_indices.size > 0:
                    outside_lats = lats[outside_indices]
                    outside_lons = lons[outside_indices]
                else:
                    outside_lats = np.array([])
                    outside_lons = np.array([])

                log.info("percent outside valid range %.2f", percent_outside)

                # find fill values -------------------------------------------------------------
                if data_set.get("fill_value") is not None:
                    log.info("finding fill_value %s", str(data_set.get("fill_value")))
                    fv_vals_bool = vals == data_set["fill_value"]
                    percent_fv = np.mean(fv_vals_bool) * 100.0
                else:
                    percent_fv = 0.0
                    fv_vals_bool = np.full_like(vals, False, bool)

                log.info("percent FV %.2f", percent_fv)

                fv_indices = np.where(fv_vals_bool)[0]
                if fv_indices.size > 0:
                    fv_lats = lats[fv_indices]
                    fv_lons = lons[fv_indices]
                else:
                    fv_lats = np.array([])
                    fv_lons = np.array([])

                valid_vals_bool = ~nan_vals_bool & ~fv_vals_bool & ~outside_vals_bool

                valid_indices = np.where(valid_vals_bool)[0]
                if valid_indices.size > 0:
                    vals = vals[valid_indices]
                    lats = lats[valid_indices]
                    lons = lons[valid_indices]
                else:
                    vals = np.array([])
                    lats = np.array([])
                    lons = np.array([])

                percent_valid = np.mean(valid_vals_bool) * 100.0

                # ------------------------------------------------------------------------------
                # Plot data
                # ------------------------------------------------------------------------------

                if valid_indices.size > 0:
                    # Get colormap info for this dataset
                    cmap_info = {
                        "cmap_name": data_set.get("cmap_name", self.thisarea.cmap_name),
                        "cmap_over_color": data_set.get(
                            "cmap_over_color", self.thisarea.cmap_over_color
                        ),
                        "cmap_under_color": data_set.get(
                            "cmap_under_color", self.thisarea.cmap_under_color
                        ),
                        "cmap_extend": data_set.get("cmap_extend", self.thisarea.cmap_extend),
                        "min_plot_range": data_set.get("min_plot_range", np.nanmin(vals)),
                        "max_plot_range": data_set.get("max_plot_range", np.nanmax(vals)),
                    }

                    if is_flag_data:
                        # Plot flag values
                        self.plot_flag_data(
                            fig,
                            ax,
                            lats,
                            lons,
                            vals,
                            data_set,
                        )
                    else:
                        # Plot normal values
                        scatter, cmap = self.plot_data(
                            ax,
                            lats,
                            lons,
                            vals,
                            cmap_info,
                            data_set.get("plot_size_scale_factor", 1.0),
                            data_set.get("plot_alpha", 1.0),
                        )

                # Only draw colorbar and histograms of 1st data set
                if ds_num == 0:
                    #  draw mini-map of bad values
                    self.draw_minimap_bad(
                        percent_valid,
                        nan_lats,
                        nan_lons,
                        percent_nan,
                        fv_lats,
                        fv_lons,
                        percent_fv,
                        outside_lats,
                        outside_lons,
                        percent_outside,
                        data_set,
                    )

                    ds_name_0 = data_set.get("name", "unnamed")
                    if valid_indices.size > 0 and not is_flag_data:
                        if self.thisarea.draw_colorbar:
                            cbar = self.draw_colorbar(
                                data_set,
                                fig,
                                scatter,
                                data_set.get("name", "unnamed"),
                                data_set.get("units", "no units"),
                            )

                            self.draw_stats(cbar, vals)

                        if self.thisarea.show_histograms:
                            self.draw_histograms(
                                fig,
                                vals,
                                data_set.get("min_plot_range", np.nanmin(vals)),
                                data_set.get("max_plot_range", np.nanmax(vals)),
                                data_set.get("units", "no units"),
                                cmap,
                            )

                        if self.thisarea.show_latitude_scatter:
                            self.draw_latitude_vs_vals_plot(
                                fig,
                                vals,
                                lats,
                                data_set.get("name", "unnamed"),
                                data_set.get("units", "no units"),
                            )

        # ----------------------------------------------------------------------------------------
        # Optionally overlay a hillshade layer
        # ----------------------------------------------------------------------------------------

        if self.thisarea.apply_hillshade_to_vals:
            log.info("Applying hillshade effect to plotted parameter...")

            Background("hillshade", self.thisarea).load(
                ax,
                dataprj,
                hillshade_params=self.thisarea.hillshade_params,
                zorder=21,
            )

        # ----------------------------------------------------------------------------------------
        # Overlay coastlines
        # ----------------------------------------------------------------------------------------
        print("drawing coastline for main map..")

        use_cartopy_coastline = self.thisarea.use_cartopy_coastline
        self.draw_coastlines(
            ax,
            dataprj,
            self.thisarea.coastline_color,
            self.thisarea.draw_coastlines,
            use_antarctica_medium_coastline=self.thisarea.use_antarctica_medium_coastline,
            use_cartopy_coastline=use_cartopy_coastline,
        )

        # ----------------------------------------------------------------------------------------
        # Overlay latitude, longitude grid lines as per area specification
        # ----------------------------------------------------------------------------------------

        self.draw_gridlines(
            ax,
            self.thisarea.show_gridlines,
            self.thisarea.gridline_color,
            circle,
            draw_gridlabels=self.thisarea.draw_gridlabels,
            gridlabel_color=self.thisarea.gridlabel_color,
            inner_gridlabel_color=self.thisarea.inner_gridlabel_color,
        )

        # ----------------------------------------------------------------------------------------
        #   Draw map scale bar
        # ----------------------------------------------------------------------------------------

        self.draw_mapscale_bar(ax, dataprj)

        # -----------------------------------------------------------------------------------------
        # Show page
        # -----------------------------------------------------------------------------------------
        if output_dir or output_file:
            if output_dir and output_file:
                plot_filename = f"{output_dir}/{output_file}"
            elif output_file and not output_dir:
                plot_filename = output_file
            elif output_dir and not output_file:
                plot_filename = f"{output_dir}/param_{ds_name_0}_{self.area}.png"
            if ".png" != plot_filename[-4:]:
                plot_filename += ".png"
            log.info("Saving plot to %s at %d dpi", plot_filename, dpi)
            plt.savefig(plot_filename, dpi=dpi, transparent=transparent_background)

        else:
            plt.show()
            plt.close()

    def draw_stats(self, cbar, vals: np.ndarray):
        """plot stats info (min,max,mean,std,MAD,nvals) of vals
           positioned around colorbar axes

        Args:
            cbar (Axes): colorbar axes instance
            vals (np.ndarray): values array (after Nan filtering) used to calculate and draw stats
                               info
        """
        # Step 1: Get the colorbar's bounding box in figure coordinates
        bbox = cbar.ax.get_window_extent().transformed(plt.gcf().transFigure.inverted())

        if self.thisarea.colorbar_orientation == "vertical":
            # Step 2: Calculate positions for the text for a vertical colorbar
            # Adjust these values as needed for your specific figure layout
            offset_y = 0.005  # Vertical offset from the colorbar ends
            text_bottom_y = bbox.y0 - offset_y
            text_top_y = bbox.y1 + offset_y
            text_x = bbox.x0 + (bbox.width / 2)  # Horizontally centered

            # Step 3: Add text to the top and bottom of the colorbar
            min_str = r"$\bf{min} $" + f"={np.min(vals):.2f}"
            max_str = r"$\bf{max} $" + f"={np.max(vals):.2f}"
            mean_str = r"$\bf{mean} $" + f"={np.mean(vals):.2f}"
            median_str = r"$\bf{median} $" + f"={np.median(vals):.2f}"
            std_str = r"$\bf{stdev} $" + f"={np.std(vals):.2f}"
            mad_str = r"$\bf{MAD} $" + f"={calculate_mad(vals):.2f}"
            nvals_str = r"$\bf{nvals} $" + f"={len(vals)}"

            plt.gcf().text(text_x, text_bottom_y, min_str, ha="center", va="top")
            plt.gcf().text(text_x, text_top_y, max_str, ha="center", va="bottom")
            text_x -= 0.05
            yoffset = 0.025
            text_top_y += 0.04
            plt.gcf().text(
                text_x + self.thisarea.stats_position_x_offset,
                text_top_y + self.thisarea.stats_position_y_offset,
                mean_str,
                ha="left",
                va="bottom",
            )
            text_top_y += yoffset
            plt.gcf().text(
                text_x + self.thisarea.stats_position_x_offset,
                text_top_y + self.thisarea.stats_position_y_offset,
                median_str,
                ha="left",
                va="bottom",
            )
            text_top_y += yoffset
            plt.gcf().text(
                text_x + self.thisarea.stats_position_x_offset,
                text_top_y + self.thisarea.stats_position_y_offset,
                std_str,
                ha="left",
                va="bottom",
            )
            text_top_y += yoffset
            plt.gcf().text(
                text_x + self.thisarea.stats_position_x_offset,
                text_top_y + self.thisarea.stats_position_y_offset,
                mad_str,
                ha="left",
                va="bottom",
            )
            text_top_y += yoffset
            plt.gcf().text(
                text_x + self.thisarea.stats_position_x_offset,
                text_top_y + self.thisarea.stats_position_y_offset,
                nvals_str,
                ha="left",
                va="bottom",
            )

        else:  # horizontal colorbar
            # Step 2: Calculate positions for the text
            # Adjust these values as needed for your specific figure layout
            offset_x = 0.005  # Horizontal offset from the colorbar ends
            text_left_x = bbox.x0 - offset_x
            text_right_x = bbox.x1 + offset_x
            text_y = bbox.y0 + (bbox.height / 2)  # Vertically centered

            # Step 3: Add text to the left and right of the colorbar
            min_str = f"min:{np.min(vals):.2f}"
            if len(min_str) > 11:
                min_str = f"{np.min(vals):.2f}"
            max_str = f"min:{np.max(vals):.2f}"
            if len(max_str) > 11:
                max_str = f"{np.max(vals):.2f}"
            plt.gcf().text(text_left_x, text_y, min_str, ha="right", va="center")
            plt.gcf().text(text_right_x, text_y, max_str, ha="left", va="center")
            text_y += 0.04
            text_left_x -= 0.04

            plt.gcf().text(
                text_left_x + self.thisarea.stats_position_x_offset,
                text_y + self.thisarea.stats_position_y_offset,
                r"$\bf{MAD}: $" + f"{calculate_mad(vals):.2f}",
                ha="left",
                va="center",
            )
            text_y += 0.02
            plt.gcf().text(
                text_left_x + self.thisarea.stats_position_x_offset,
                text_y + self.thisarea.stats_position_y_offset,
                r"$\bf{mean}: $" + f"{np.mean(vals):.2f}",
                ha="left",
                va="center",
            )
            text_y += 0.02
            plt.gcf().text(
                text_left_x + self.thisarea.stats_position_x_offset,
                text_y + self.thisarea.stats_position_y_offset,
                r"$\bf{median}: $" + f"{np.median(vals):.2f}",
                ha="left",
                va="center",
            )
            text_y += 0.02
            plt.gcf().text(
                text_left_x + self.thisarea.stats_position_x_offset,
                text_y + self.thisarea.stats_position_y_offset,
                r"$\bf{std}: $" + f"{np.std(vals):.2f}",
                ha="left",
                va="center",
            )
            text_y += 0.02
            plt.gcf().text(
                text_left_x + self.thisarea.stats_position_x_offset,
                text_y + self.thisarea.stats_position_y_offset,
                r"$\bf{nvals}: $" + f"{len(vals)}",
                ha="left",
                va="center",
            )

    def draw_minimap_bad(
        self,
        percent_valid: float,
        nan_lats: np.ndarray,
        nan_lons: np.ndarray,
        percent_nan: float,
        fv_lats: np.ndarray,
        fv_lons: np.ndarray,
        percent_fv: float,
        outside_lats: np.ndarray,
        outside_lons: np.ndarray,
        percent_outside: float,
        dataset_params: dict,
    ):
        """draw a minimap to show Nan, FV and out of range values

        Args:
            percent_valid (float): percent of valid data in area
            nan_lats (np.ndarray): latitude locations corresponding to Nan data
            nan_lons (np.ndarray): longitude locations corresponding to Nan data
            percent_nan (float): percent of Nan values in area
            fv_lats (np.ndarray): latitude locations corresponding to FV data
            fv_lons (np.ndarray): longitude locations corresponding to FV data
            percent_fv (float): percent of Fill value values in area
            outside_lats (np.ndarray): latitude locations corresponding to out of range data
            outside_lons (np.ndarray): longitude locations corresponding to out of range data
            percent_outside (float): percent of out of range values in area
            dataset_params (dict): data set parameters
        """
        if not self.thisarea.show_bad_data_map:
            log.info("drawing show_bad_data_map disabled")
            return
        log.info("drawing show_bad_data_map")
        (
            ax_minimap,
            dataprj_minimap,
            circle_minimap,
        ) = self.setup_projection_and_extent(self.thisarea.bad_data_minimap_axes, global_view=False)
        Background("basic_land", self.thisarea).load(
            ax_minimap, dataprj_minimap, include_features=False, resolution="low"
        )

        self.draw_coastlines(
            ax_minimap,
            dataprj_minimap,
            "grey",
            draw_coastlines=True,
            use_cartopy_coastline=self.thisarea.bad_data_minimap_coastline_resolution,
            use_antarctica_medium_coastline=False,
        )

        if self.thisarea.bad_data_minimap_draw_gridlines:
            self.draw_gridlines(
                ax_minimap,
                True,
                self.thisarea.bad_data_minimap_gridlines_color,
                circle_minimap,
                draw_gridlabels=False,
                gridlabel_color=self.thisarea.gridlabel_color,
                inner_gridlabel_color=self.thisarea.inner_gridlabel_color,
                latitude_lines=self.thisarea.bad_data_latitude_lines,
                longitude_lines=self.thisarea.bad_data_longitude_lines,
                zorder=0,
                for_minimap=True,
            )

        # plot Nan values
        if nan_lons.size > 0:
            ax_minimap.scatter(
                nan_lons,
                nan_lats,
                marker=".",
                c="r",
                s=36
                * dataset_params.get(
                    "bad_data_minimap_val_scalefactor",
                    self.thisarea.bad_data_minimap_val_scalefactor,
                ),
                transform=ccrs.PlateCarree(),
                label=f"Nan {percent_nan:.2f}%",
            )

        # plot FV values
        if fv_lons.size > 0:
            ax_minimap.scatter(
                fv_lons,
                fv_lats,
                marker=".",
                c="orange",
                s=36
                * dataset_params.get(
                    "bad_data_minimap_val_scalefactor",
                    self.thisarea.bad_data_minimap_val_scalefactor,
                ),
                transform=ccrs.PlateCarree(),
                label=f"FV {percent_fv:.2f}%",
            )

        # plot outside range values
        if outside_lons.size > 0:
            ax_minimap.scatter(
                outside_lons,
                outside_lats,
                marker=".",
                c="pink",
                s=36
                * dataset_params.get(
                    "bad_data_minimap_val_scalefactor",
                    self.thisarea.bad_data_minimap_val_scalefactor,
                ),
                transform=ccrs.PlateCarree(),
                label=f"<|> {percent_outside:.2f}%",
            )
        if (nan_lons.size == 0) and (outside_lons.size == 0) and (fv_lons.size == 0):
            ax_minimap.scatter([], [], marker=".", s=1, label="Bad Data")
        ax_minimap.legend(
            loc="upper right", bbox_to_anchor=self.thisarea.bad_data_minimap_legend_pos
        )

        ax_minimap.text(
            0.44,
            1.05,
            f"Valid in area: {percent_valid:.2f}% ",
            fontsize=9,
            transform=ax_minimap.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )

    def draw_minimap(
        self,
    ):
        """draw a minimap to show Nan, FV and out of range values

        Args:

        """
        if not self.thisarea.show_minimap:
            log.info("drawing mini-map disabled")
            return
        log.info("drawing mini-map")
        (
            ax_minimap,
            dataprj_minimap,
            circle_minimap,
        ) = self.setup_projection_and_extent(self.thisarea.minimap_axes, global_view=True)
        Background("basic_land", self.thisarea).load(
            ax_minimap, dataprj_minimap, include_features=False, resolution="low"
        )

        self.draw_coastlines(
            ax_minimap,
            dataprj_minimap,
            "grey",
            draw_coastlines=True,
            use_cartopy_coastline="low",
            use_antarctica_medium_coastline=False,
        )

        if self.thisarea.minimap_draw_gridlines:
            if self.thisarea.hemisphere == "north":
                self.draw_gridlines(
                    ax_minimap,
                    True,
                    "black",
                    circle_minimap,
                    draw_gridlabels=False,
                    gridlabel_color=self.thisarea.gridlabel_color,
                    inner_gridlabel_color=self.thisarea.inner_gridlabel_color,
                    latitude_lines=[50, 70],
                    longitude_lines=[0, 60, 120, 180, -120, -60],
                    zorder=0,
                    for_minimap=True,
                )
            else:
                self.draw_gridlines(
                    ax_minimap,
                    True,
                    "black",
                    circle_minimap,
                    draw_gridlabels=False,
                    gridlabel_color=self.thisarea.gridlabel_color,
                    inner_gridlabel_color=self.thisarea.inner_gridlabel_color,
                    latitude_lines=[-50, -70],
                    longitude_lines=[0, 60, 120, 180, -120, -60],
                    zorder=0,
                    for_minimap=True,
                )
        if self.thisarea.minimap_circle:
            circle_color = self.thisarea.minimap_circle[3]
            projx1, projy1 = dataprj_minimap.transform_point(
                self.thisarea.minimap_circle[1],
                self.thisarea.minimap_circle[0],
                ccrs.Geodetic(),
            )  # get proj coord of (lon,lat)
            ax_minimap.add_patch(
                mpatches.Circle(
                    xy=(projx1, projy1),
                    radius=self.thisarea.minimap_circle[2],
                    color=circle_color,
                    alpha=0.3,
                    transform=dataprj_minimap,
                    zorder=30,
                )
            )

    def draw_latitude_vs_vals_plot(
        self, fig: Figure, vals: np.ndarray, lats: np.ndarray, varname: str, units: str
    ):
        """plot latitude vs vals

        Args:
            fig (Figure): plot figure
            vals (np.ndarray): values to be plotted
            lats (np.ndarray): latitude values in degrees
            varname (str): name of data set
            units (str): units of data set
        """

        lat_axes = fig.add_axes(
            self.thisarea.latvals_axes,
        )  # left, bottom, width, height (fractions of axes)
        lat_axes.scatter(lats.tolist(), vals, marker="D", s=0.1)
        lat_axes.set_ylabel(f"{varname} ({units})")
        lat_axes.set_xlabel("Latitude (degs)")

    def draw_histograms(
        self,
        fig,
        vals: np.ndarray,
        min_plot_range: float,
        max_plot_range: float,
        varunits: str,
        cmap,
    ):
        """draw two histograms of plot range and full range

        Args:
            fig (Figure): plot figure
            vals (np.ndarray): values to be histogrammed
            min_plot_range (float): minimum plot range
            max_plot_range (float): maximum plot range
            varunits (str): units of vals
            cmap (_type_): colormap instance
        """

        if len(vals) < 2:
            log.error("not enough values to create histogram")
            return
        if np.nanmin(vals) == np.nanmax(vals):
            log.error("can't create histogram from equal values")
            return

        hist_axes = fig.add_axes(
            self.thisarea.histogram_plotrange_axes
        )  # left, bottom, width, height (fractions of axes)

        _, bins, patches = hist_axes.hist(
            np.array(vals),
            120,
            range=[min_plot_range, max_plot_range],
            density=1,
            facecolor="darkblue",
            alpha=0.75,
            orientation="horizontal",
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, "facecolor", cmap(c))

        hist_axes.set_xticklabels([])
        hist_axes.get_xaxis().set_visible(False)
        hist_axes.set_facecolor("white")

        hist_axes.text(0.05, -0.05, "Plot Range", fontsize=10, transform=hist_axes.transAxes)

        # -----------------------------------------------------------------------------------------
        # Histogram (full range) in single color
        # -----------------------------------------------------------------------------------------
        hist_axes = fig.add_axes(
            self.thisarea.histogram_fullrange_axes
        )  # left, bottom, width, height (fractions of axes)

        _, bins, patches = hist_axes.hist(
            vals,
            120,
            range=[np.nanmin(vals), np.nanmax(vals)],
            density=True,
            facecolor="darkblue",
            alpha=0.75,
            orientation="horizontal",
        )

        hist_axes.set_xticklabels([])
        hist_axes.get_xaxis().set_visible(False)
        hist_axes.set_facecolor("white")
        hist_axes.text(0.05, -0.05, "Full Range", fontsize=10, transform=hist_axes.transAxes)
        hist_axes.set_ylabel(f"({varunits})")

    def draw_mapscale_bar(self, ax, dataprj):
        """draw the map scale bar in km

        Args:
            ax (GeoAxis): the main map plot axis
            dataprj (): the data projection
        """

        print("Adding map scale bar")

        mapscale = self.thisarea.mapscale

        # Centre point of scale bar in data coordinates (m)
        cx, cy = self.thisarea.latlon_to_xy(mapscale[1], mapscale[0])

        print(
            "Scalebar location latlon == x,y",
            mapscale[0],
            mapscale[1],
            cx,
            cy,
        )

        print("Scalebar width", mapscale[4])
        cx0 = cx - (mapscale[4] * 1e3) / 2.0
        cx1 = cx + (mapscale[4] * 1e3) / 2.0

        ax.plot(
            [cx0, cx1],
            [cy, cy],
            color=mapscale[5],
            linewidth=2,
            marker="|",
            markersize=9,
            transform=dataprj,
            zorder=30,
        )

        ax.text(
            cx0 + (cx1 - cx0) / 3,
            cy + mapscale[6] * 1e3,
            "km",
            transform=dataprj,
            color=mapscale[5],
            fontsize=9,
            zorder=30,
        )

        ax.text(
            cx0 + (cx1 - cx0) / 3,
            cy - 2 * mapscale[6] * 1e3,
            f"{mapscale[4]}",
            transform=dataprj,
            color=mapscale[5],
            fontsize=9,
            zorder=30,
        )

    def draw_colorbar(self, dataset: dict, fig, scatter, varname: str, varunits: str):
        """draw the colorbar

        Args:
            dataset (dict): the data set dict
            fig (Figure): the plot figure
            scatter (_type_): _description_
            varname (str): name of data set
            varunits (str): units of data set
        Returns:
            colorbar_axes(Axes): Axes instance for the colorbar
        """

        if self.thisarea.colorbar_orientation == "vertical":
            colorbar_axes = fig.add_axes(
                self.thisarea.vertical_colorbar_axes
            )  # left, bottom, width, height (fractions of axes)
            cbar = fig.colorbar(
                scatter,
                cax=colorbar_axes,
                orientation="vertical",
                extend=dataset.get("cmap_extend", self.thisarea.cmap_extend),
            )
        else:  # horizontal colorbar
            colorbar_axes = fig.add_axes(
                self.thisarea.horizontal_colorbar_axes
            )  # left, bottom, width, height (fractions of axes)
            cbar = fig.colorbar(
                scatter,
                cax=colorbar_axes,
                orientation="horizontal",
                extend=dataset.get("cmap_extend", self.thisarea.cmap_extend),
            )
        cbar.set_label(f"{varname} ({varunits})")

        return cbar

    def plot_flag_data(
        self,
        fig,
        ax,
        lats,
        lons,
        vals,
        data_set,
    ):
        """plot flag data

        Args:
            fig (Figure) : the plot figure
            ax (GeoAxesSubplot): the main plot axis
            lats (np.ndarray): latitude values
            lons (np.ndarray): longitude values
            vals (np.ndarray): data containing flag values to plot
            data_set (dict): the current data set
        """
        varname = data_set.get("name", "unnamed")
        flag_names = data_set.get("flag_names", [])
        flag_values = data_set.get("flag_values", [])
        flag_colors = data_set.get("flag_colors", get_unique_colors(len(flag_names)))
        if len(flag_colors) != len(flag_names):
            log.info("Generating unique flag colors")
            flag_colors = get_unique_colors(len(flag_names))

        plot_size_scale_factor = data_set.get("plot_size_scale_factor", 1.0)

        log.info("plotting flag data")
        log.info("flag_names: %s", flag_names)
        log.info("flag_values: %s", str(flag_values))
        log.info("flag_colors: %s", str(flag_colors))

        if len(flag_names) < 1:
            log.error("zero length flag_names found")
            raise ValueError("zero length flag_names found")
        if len(flag_names) != len(flag_values):
            log.error(
                "flag_names[%d] and flag_values[%d] have different lengths",
                len(flag_names),
                len(flag_values),
            )
            return
        if len(flag_names) != len(flag_colors):
            log.error(
                "flag_names[%d] and flag_colors[%d] have different lengths!",
                len(flag_names),
                len(flag_colors),
            )
            return

        # Default size is 36. Scale up or down
        scale_factor = 36 * plot_size_scale_factor

        number_of_each_flag = []
        percent_of_each_flag = []
        total = vals.size

        for flag_index in range(len(flag_names)):
            flagindices = np.flatnonzero(vals == flag_values[flag_index])
            if flagindices.size > 0:
                ax.scatter(
                    lons[flagindices],
                    lats[flagindices],
                    marker=".",
                    c=flag_colors[flag_index],
                    s=scale_factor,
                    transform=ccrs.PlateCarree(),
                    zorder=20,
                )

            number_of_each_flag.append(flagindices.size)
            if total > 0:
                percent_of_each_flag.append(100.0 * flagindices.size / total)
            else:
                percent_of_each_flag.append(0.0)
        number_of_flags = flag_index

        if self.thisarea.include_flag_legend:
            # Add a legend. Firstly create new handles to display legend as wider bar
            handles = []
            for flag_index, flagname in enumerate(flag_names):
                if flag_colors is None:
                    h = mpatches.Patch(label=flagname)
                else:
                    h = mpatches.Patch(color=flag_colors[flag_index], label=flagname)
                handles.append(h)

            if self.thisarea.flag_legend_xylocation[0] is not None:
                plt.legend(
                    handles=handles,
                    loc="lower right",
                    bbox_to_anchor=(
                        self.thisarea.flag_legend_xylocation[0],
                        self.thisarea.flag_legend_xylocation[1],
                    ),
                )
            else:
                plt.legend(handles=handles, loc=self.thisarea.flag_legend_location)

        if self.thisarea.include_flag_percents:
            # Flag bar plot is composed of 2 main sections
            #   1) a colour square per flag, (similar to a legend, so you can see flag colours
            #    that have low percentages)
            #   2) a percent bar indicator
            #   3) percent number labels

            # Setup colour square y-axis
            # [left,bottom, width, height] of axis.
            axis_flag_color_square = fig.add_axes(
                [
                    self.thisarea.flag_perc_axis[0],
                    self.thisarea.flag_perc_axis[1],
                    0.02,
                    0.04 * number_of_flags,
                ],  # left, bottom, width, height (fractions of axes)
            )
            axis_flag_color_square.spines["top"].set_visible(False)
            axis_flag_color_square.spines["right"].set_visible(False)
            axis_flag_color_square.spines["bottom"].set_visible(False)
            axis_flag_color_square.spines["left"].set_visible(False)
            axis_flag_color_square.get_xaxis().set_ticks([])
            axis_flag_color_square.get_yaxis().set_ticks([])

            # Plot bar plot of number of each flag
            bar_width = 0.8
            for flag_index, flagname in enumerate(flag_names):
                axis_flag_color_square.barh(
                    flag_index,
                    10,
                    bar_width,
                    color=None if flag_colors is None else flag_colors[flag_index],
                )
            plt.xlabel("")  # bottom label
            plt.yticks(np.arange(len(flag_names)), flag_names)  # left hand axis labels

            # ----------------------------------------------------------------------------------
            # scale bar plot depending upon number of flags
            flag_bar_ax = fig.add_axes(
                [
                    self.thisarea.flag_perc_axis[0],
                    self.thisarea.flag_perc_axis[1],
                    self.thisarea.flag_perc_axis[2],
                    0.04 * number_of_flags,
                ]
            )  # left, bottom, width, height (fractions of axes)

            # Plot bar plot of number of each flag

            for flag_index, flagname in enumerate(flag_names):
                flag_bar_ax.barh(
                    flag_index,
                    percent_of_each_flag[flag_index],
                    bar_width,
                    color=None if flag_colors is None else flag_colors[flag_index],
                )

            plt.yticks(np.arange(len(flag_names)), flag_names)  # left hand axis labels
            plt.xlabel("% " + varname)  # bottom label
            plt.xlim(left=-30, right=100)  # x-axis range limits
            flag_bar_ax.patch.set_alpha(0.01)

            # Setup second y-axis on right with % of flag numbers as tick marks
            flag_bar_ax2 = flag_bar_ax.twinx()
            sperc = []
            for flag_index, flagname in enumerate(flag_names):
                flag_bar_ax2.barh(
                    flag_index,
                    percent_of_each_flag[flag_index],
                    bar_width,
                    color=None if flag_colors is None else flag_colors[flag_index],
                )
                sperc.append(f"{percent_of_each_flag[flag_index]:.2f}%")

            plt.yticks(np.arange(len(flag_names)), sperc)

            # plot a dividing line between colour squares and colour bars
            plt.axvline(x=0, color="lightgray", linestyle="-", lw="2")

    def plot_data(
        self,
        ax: GeoAxesSubplot,
        lats: np.ndarray,
        lons: np.ndarray,
        vals: np.ndarray,
        cmap_info: dict,
        plot_size_scale_factor=1.0,
        plot_alpha=1.0,
    ):
        """plot lat,lon,vals, data on map

        Args:
            ax (GeoAxesSubplot): the main plot axis
            lats (np.ndarray): latitude values
            lons (np.ndarray): longitude values
            vals (np.ndarray): values to plot
            cmap_info: (dict): colormap info
        """

        # load colormap
        new_cmap = plt.colormaps[cmap_info["cmap_name"]].copy()
        if cmap_info["cmap_over_color"] is not None:
            new_cmap.set_over(cmap_info["cmap_over_color"])
        if cmap_info["cmap_under_color"] is not None:
            new_cmap.set_under(cmap_info["cmap_under_color"])

        if cmap_info["min_plot_range"] is not None:
            vmin = cmap_info["min_plot_range"]
        else:
            vmin = np.nanmin(vals)

        if cmap_info["max_plot_range"] is not None:
            vmax = cmap_info["max_plot_range"]
        else:
            vmax = np.nanmax(vals)
        # Normalizer
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Default size is 36. Scale up or down
        scale_factor = 36 * plot_size_scale_factor

        scatter = None
        try:
            scatter = ax.scatter(
                lons,
                lats,
                marker=".",
                c=vals,
                cmap=new_cmap,
                norm=norm,
                s=scale_factor,
                alpha=plot_alpha,
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
        except StopIteration:
            # Can be thrown if vals is entirely nan
            # Catch and substitute a blank plot so scripts can continue
            log.error("Error: Requested a plot with entirely NaN data")
            scatter = ax.scatter(
                lons,
                lats,
                marker=".",
                c=np.ones_like(vals),
                # cmap=thiscmap,
                s=0,
                # vmin=minrange,
                # vmax=maxrange,
                alpha=0.0,
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
        return scatter, new_cmap

    def draw_gridlines(
        self,
        ax: GeoAxesSubplot,
        show_gridlines: bool,
        gridline_color: str,
        circle,
        draw_gridlabels: bool,
        gridlabel_color: str,
        inner_gridlabel_color: str,
        longitude_lines=None,
        latitude_lines=None,
        zorder=30,
        for_minimap=False,
    ):
        """draw latitude and longitude grid lines on maps

        Args:
            ax (GeoAxesSubplot): cartopy axis
            show_gridlines (bool):
            gridline_color (str): color of gridlines
            circle (_type_): _description_
            draw_gridlabels (bool, optional): _description_. Defaults to True.
            gridlabel_color (str) : color of grid labels
            inner_gridlabel_color (str) : color of inner grid labels
            longitude_lines (List[float]|None, optional): longitude positions for grid lines.
            latitude_lines (List[float]|None, optional): latitude positions for grid lines.
            zorder (int, optional): vertical order. Defaults to 10.
            for_minimap (bool, optional): if used for mini-map. Defaults to False.
        """

        if not show_gridlines:
            return

        log.info("draw grid lines..")

        if not latitude_lines:
            latitude_lines = self.thisarea.latitude_gridlines
        if not longitude_lines:
            longitude_lines = self.thisarea.longitude_gridlines

        # -----------------------------------------------------------------------------------------
        #  Overlay gridlines
        # -----------------------------------------------------------------------------------------

        # draw meridians and parallels
        gl = ax.gridlines(
            color=gridline_color,
            linestyle=(0, (1, 1)),
            xlocs=longitude_lines,
            ylocs=latitude_lines,
            zorder=zorder,
        )
        gl.xlabel_style = {
            "color": gridlabel_color,
            "size": self.thisarea.gridlabel_size,
        }
        gl.ylabel_style = {
            "color": gridlabel_color,
            "size": self.thisarea.gridlabel_size,
        }
        gl.n_steps = 90
        # [left,right,top,bottom]
        if draw_gridlabels and not self.thisarea.round:
            gl.left_labels = self.thisarea.labels_at_left
            gl.right_labels = self.thisarea.labels_at_right
            gl.top_labels = self.thisarea.labels_at_top
            gl.bottom_labels = self.thisarea.labels_at_bottom
            gl.x_inline = gl.y_inline = False
            gl.draw_labels = True

        if self.thisarea.round:
            # Southern hemisphere
            if self.thisarea.hemisphere == "south":
                latitude_position = -58
                if self.thisarea.latitude_of_radial_labels:
                    latitude_position = self.thisarea.latitude_of_radial_labels
                latitude_adjust = [-3.7, -3.8, -3.4, -3, -2.6]
                if draw_gridlabels:
                    for i, lon in enumerate(range(0, 180, 40)):
                        ax.text(
                            lon,
                            latitude_position + latitude_adjust[i],
                            str(lon) + "E",
                            color=gridlabel_color,
                            size=self.thisarea.gridlabel_size,
                            transform=ccrs.PlateCarree(),
                        )
                    latitude_adjust = [-2.4, -1.8, -1.4, -2.2]
                    for i, lon in enumerate(range(-40, -180, -40)):
                        ax.text(
                            lon,
                            latitude_position + latitude_adjust[i],
                            str(-1 * lon) + "W",
                            color=gridlabel_color,
                            size=self.thisarea.gridlabel_size,
                            transform=ccrs.PlateCarree(),
                        )
                    ax.text(
                        -162,
                        -66,
                        "-66",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.text(
                        -162,
                        -70,
                        "-70",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.text(
                        -162,
                        -74,
                        "-74",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size,
                        transform=ccrs.PlateCarree(),
                    )

                    ax.text(
                        -162,
                        -88,
                        "-88",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size,
                        transform=ccrs.PlateCarree(),
                    )

            else:  # Northern hemisphere round plots
                if draw_gridlabels:
                    # Set the latitude position of the labels for round plots
                    latitude_position = 58  # for Area.area == 'arctic'
                    if "_wide" in self.thisarea.name:
                        latitude_position = 42
                    latitude_adjust = [
                        0.8,
                        0.9,
                        0.9,
                        0.9,
                        0.1,
                    ]  # label position adjustment

                    lon_enum = [0, 40, 80, 120, 160]
                    if self.thisarea.lon_0 is not None and self.thisarea.lon_0 == 0.0:
                        latitude_adjust = [
                            0.2,
                            0.8,
                            0.8,
                            0.8,
                            0.4,
                        ]  # label position adjustment
                        lon_enum = [0, 40, 80, 120, 160]

                    for i, lon in enumerate(lon_enum):
                        ax.text(
                            lon,
                            latitude_position + latitude_adjust[i],
                            str(lon) + "E",
                            color=gridlabel_color,
                            size=self.thisarea.gridlabel_size,
                            transform=ccrs.PlateCarree(),
                        )

                    latitude_adjust = [0.4, -0.6, -2.0, -1.4]
                    lon_enum = [-40, -80, -120, -160]
                    if self.thisarea.lon_0 is not None and self.thisarea.lon_0 == 0.0:
                        latitude_adjust = [-1, -1.5, -1.5, -0.4]
                        lon_enum = [-40, -80, -120, -160]
                    for i, lon in enumerate(lon_enum):
                        ax.text(
                            lon,
                            latitude_position + latitude_adjust[i],
                            str(-1 * lon) + "W",
                            color=gridlabel_color,
                            size=self.thisarea.gridlabel_size,
                            transform=ccrs.PlateCarree(),
                        )
                    # Add annotations for 70N and 80N latitude
                    ax.text(
                        0,
                        69,
                        "70N",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size - 1,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.text(
                        0,
                        79,
                        "80N",
                        color=inner_gridlabel_color,
                        size=self.thisarea.inner_gridlabel_size - 1,
                        transform=ccrs.PlateCarree(),
                    )

            # Draw a surrounding circle
            patch = mpatches.PathPatch(
                circle,
                facecolor="none",
                edgecolor="lightgrey",
                lw=4,
                transform=ax.transAxes,
                zorder=zorder,
            )

            ax.add_patch(patch)

            if not for_minimap:
                theta = np.linspace(0, 2 * np.pi, 100)
                center, radius = [0.5, 0.5], 0.497
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                patch = mpatches.PathPatch(
                    circle,
                    facecolor="none",
                    edgecolor="slategrey",
                    lw=1,
                    transform=ax.transAxes,
                    zorder=zorder,
                )
                ax.add_patch(patch)

    def draw_coastlines(
        self,
        ax: GeoAxesSubplot,
        dataprj,
        coastline_color: str,
        draw_coastlines: bool,
        use_cartopy_coastline: str,
        use_antarctica_medium_coastline: bool,
    ):
        """draw coastlines over map

        Args:
            ax (GeoAxesSubplot): matplotlib Axis
            dataprj (_type_): current cartopy crs
            coastline_color (str): coastline color to use
            draw_coastlines (bool):  draw coastline or not
            use_cartopy_coastline (str): 'no','low','medium','high'.
            use_antarctica_medium_coastline (bool): use antarctic coastline (including iceshelves)
        """

        # Check if coastline drawing is required
        if draw_coastlines is not None:
            if not draw_coastlines:
                return

        # Antarctic Medium Coastline

        if use_antarctica_medium_coastline:
            print("Plotting Antarctic Medium coastline..")

            fname = (
                os.environ["CPOM_SOFTWARE_DIR"]
                + "/cpom/resources/coastlines/Coastline_medium_res_line"
                "/Coastline_medium_res_line_WGS84.shp"
            )
            sf = shp.Reader(fname)
            all_lons = []
            all_lats = []
            poly_index = []
            num_shapes = len(sf.shapeRecords())
            for j, shape in enumerate(sf.shapeRecords()):
                lons = [i[0] for i in shape.shape.points[:]]
                lats = [i[1] for i in shape.shape.points[:]]
                all_lons.extend(lons)
                all_lats.extend(lats)
                jj = [j] * len(lats)
                poly_index.extend(jj)

            x, y = self.thisarea.latlon_to_xy(all_lats, all_lons)
            _poly_index = np.array(poly_index)
            x = np.array(x)
            y = np.array(y)

            for i in range(num_shapes):
                thispoly = np.flatnonzero(_poly_index == i)
                thisx = x[thispoly]
                thisy = y[thispoly]

                poly_corners = np.zeros((len(thisx), 2), np.float64)
                poly_corners[:, 0] = thisx
                poly_corners[:, 1] = thisy

                poly = mpatches.Polygon(
                    poly_corners,
                    closed=False,
                    edgecolor=coastline_color,
                    fill=False,
                    lw=1,
                    ls="-",
                    transform=dataprj,
                    zorder=101,
                )
                ax.add_patch(poly)

        # Cartopy Coastline

        if use_cartopy_coastline == "low":
            ax.coastlines(resolution="110m", color=coastline_color)
        elif use_cartopy_coastline == "medium":
            ax.coastlines(resolution="50m", color=coastline_color)
        elif use_cartopy_coastline == "high":
            ax.coastlines(resolution="10m", color=coastline_color)

    def draw_area_polygon_mask(
        self,
        ax: GeoAxesSubplot,
        override_mask_display,
        override_mask_color,
        dataprj,
        fill=False,
        linestyle="-",
        linecolor="red",
        linewidth=2,
    ):
        """if area has a data mask defined by one or more polygons, draw these on map

        Args:
            ax (GeoAxesSubplot): cartopy axis
            override_mask_display (bool): if set to True (show polygon mask) or False (do not show
                                          polygon mask), overrides default for area
            override_mask_color (bool): set to a color string to override default polygon mask color
                                        for area
            dataprj (_type_): crs returned by self.setup_projection_and_extent()
            fill (bool, optional): fill polygon if True. Defaults to False.
            linestyle (str, optional): line style to use for polygon edges. Defaults to "-".
            linecolor (str, optional): line color to use for polygon edges. Defaults to "red".
            linewidth (int, optional): line width to use for polygon edges. Defaults to 2.
        """

        polygon_color = "red"
        if override_mask_color:
            polygon_color = override_mask_color

        # plot polygon boundaries
        display_polygon_mask = self.thisarea.show_polygon_overlay_in_main_map
        if override_mask_display is not None:
            display_polygon_mask = override_mask_display

        # form a polygon from xy limits mask. Only show xy limits polygon for global map where
        # fill is specified
        if self.thisarea.masktype == "xylimits" and display_polygon_mask and fill:
            x = [
                self.thisarea.mask.xlimits[0],
                self.thisarea.mask.xlimits[1],
                self.thisarea.mask.xlimits[1],
                self.thisarea.mask.xlimits[0],
                self.thisarea.mask.xlimits[0],
            ]
            y = [
                self.thisarea.mask.ylimits[1],
                self.thisarea.mask.ylimits[1],
                self.thisarea.mask.ylimits[0],
                self.thisarea.mask.ylimits[0],
                self.thisarea.mask.ylimits[1],
            ]

            poly_corners = np.zeros((len(x), 2), np.float64)
            poly_corners[:, 0] = x
            poly_corners[:, 1] = y

            poly = mpatches.Polygon(
                poly_corners,
                closed=True,
                edgecolor=linecolor,
                fill=fill,
                facecolor=polygon_color,
                lw=1,
                ls="-",
                transform=dataprj,
                alpha=0.5,
            )

            ax.add_patch(poly)

        elif self.thisarea.masktype == "xylimits" and display_polygon_mask:
            print("drawing xylimits mask...")
            x = [
                self.thisarea.mask.xlimits[0],
                self.thisarea.mask.xlimits[1],
                self.thisarea.mask.xlimits[1],
                self.thisarea.mask.xlimits[0],
                self.thisarea.mask.xlimits[0],
            ]
            y = [
                self.thisarea.mask.ylimits[1],
                self.thisarea.mask.ylimits[1],
                self.thisarea.mask.ylimits[0],
                self.thisarea.mask.ylimits[0],
                self.thisarea.mask.ylimits[1],
            ]

            poly_corners = np.zeros((len(x), 2), np.float64)
            poly_corners[:, 0] = x
            poly_corners[:, 1] = y

            poly = mpatches.Polygon(
                poly_corners,
                closed=True,
                edgecolor=linecolor,
                fill=fill,
                facecolor=polygon_color,
                lw=linewidth,
                ls=linestyle,
                transform=dataprj,
            )

            ax.add_patch(poly)

        elif self.thisarea.masktype == "polygon" and display_polygon_mask:
            print("draw mask polygon..")

            if self.thismask:
                print("mask found..")

                if self.thisarea.mask.polygon_lon.any():
                    print("draw single polygon..")

                    x, y = self.thisarea.latlon_to_xy(
                        self.thisarea.mask.polygon_lat, self.thisarea.mask.polygon_lon
                    )

                    poly_corners = np.zeros((len(x), 2), np.float64)
                    poly_corners[:, 0] = x
                    poly_corners[:, 1] = y

                    poly = mpatches.Polygon(
                        poly_corners,
                        closed=True,
                        edgecolor=linecolor,
                        fill=fill,
                        facecolor=polygon_color,
                        lw=linewidth,
                        ls=linestyle,
                        transform=dataprj,
                    )

                    ax.add_patch(poly)

                elif self.thismask.polygons_lon:
                    print("draw multiple polygons..")

                    n_polygons = len(self.thismask.polygons_lon)
                    for pi in range(n_polygons):
                        x, y = self.thisarea.latlon_to_xy(
                            self.thisarea.mask.polygons_lat[pi],
                            self.thisarea.mask.polygons_lon[pi],
                        )

                        poly_corners = np.zeros((len(x), 2), np.float64)
                        poly_corners[:, 0] = x
                        poly_corners[:, 1] = y

                        poly = mpatches.Polygon(
                            poly_corners,
                            closed=True,
                            edgecolor=linecolor,
                            fill=fill,
                            facecolor=polygon_color,
                            lw=linewidth,
                            ls=linestyle,
                            transform=dataprj,
                        )
                        ax.add_patch(poly)

        elif (
            self.thisarea.masktype == "grid"
            and display_polygon_mask
            and self.thisarea.grid_polygon_overlay_mask
        ):
            if self.thismask:
                if self.thisarea.mask.polygon_lon.any():
                    x, y = self.thisarea.latlon_to_xy(
                        self.thismask.polygon_lat, self.thismask.polygon_lon
                    )

                    poly_corners = np.zeros((len(x), 2), np.float64)
                    poly_corners[:, 0] = x
                    poly_corners[:, 1] = y

                    poly = mpatches.Polygon(
                        poly_corners,
                        closed=True,
                        edgecolor=linecolor,
                        fill=fill,
                        facecolor=polygon_color,
                        lw=linewidth,
                        ls=linestyle,
                        transform=dataprj,
                    )
                    ax.add_patch(poly)

                elif self.thisarea.mask.polygons_lon:
                    n_polygons = len(self.thismask.polygons_lon)

                    for pi in range(n_polygons):
                        x, y = self.thisarea.latlon_to_xy(
                            self.thismask.polygons_lat[pi],
                            self.thismask.polygons_lon[pi],
                        )
                        poly_corners = np.zeros((len(x), 2), np.float64)
                        poly_corners[:, 0] = x
                        poly_corners[:, 1] = y

                        poly = mpatches.Polygon(
                            poly_corners,
                            closed=True,
                            edgecolor=linecolor,
                            fill=fill,
                            facecolor=polygon_color,
                            lw=linewidth,
                            ls=linestyle,
                            transform=dataprj,
                        )
                        ax.add_patch(poly)

    def setup_projection_and_extent(
        self, axis_position=None, global_view=False, draw_axis_frame=True
    ):
        """Setup projection and extent for current Area

        Args:
            axis_position (List, optional): [left,bottom,width,height]. Defaults to None.
            global_view (bool, optional): _description_. Defaults to False.
            draw_axis_frame (bool, optional): _description_. Defaults to True.

        Returns:
            cartopy_geo_axis, data_projection_crs, circle

            if area has a circular border returns the circle

        """
        log.info("setup_projection_and_extent..")
        circle = None
        if not draw_axis_frame:
            plt.axis("off")

        # ------------------------------------------------------------------------------------------
        #  Setup projections for supported epsg numbers
        # ------------------------------------------------------------------------------------------

        # EPSG:3995: WGS 84 / Arctic Polar Stereographic
        if self.thisarea.epsg_number == 3995:
            dataprj = ccrs.epsg("3995")
            this_projection = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=71.0)

        # EPSG:3413: NSIDC Sea Ice Polar Stereographic North
        elif self.thisarea.epsg_number == 3413:
            dataprj = ccrs.epsg("3413")  # NSIDC Sea Ice Polar Stereographic North
            this_projection = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70.0)

        # EPSG:3031: Antarctic Polar Stereographic / WGS 84
        elif self.thisarea.epsg_number == 3031:
            dataprj = ccrs.epsg("3031")
            this_projection = ccrs.SouthPolarStereo(true_scale_latitude=-71.0)

        ax = plt.axes(axis_position, projection=this_projection)

        # -------------------------------------------
        # Setup the axis extent for this area
        # -------------------------------------------

        if global_view:  # To use for global view of area
            if self.thisarea.minimap_bounding_lat:
                bounding_lat = self.thisarea.minimap_bounding_lat
            else:
                if self.thisarea.hemisphere == "north":
                    bounding_lat = 40
                else:
                    bounding_lat = 60

            if self.thisarea.hemisphere == "north":
                if self.thisarea.lon_0 is not None and self.thisarea.lon_0 == 0.0:
                    # Note that the '-1' below is a fudge to expand the area to account for the
                    # clipping of the circular boundary
                    ax.set_extent(
                        [-180, 180, 90, 60 - 1], ccrs.PlateCarree()
                    )  # min lon, max_lon, max_lat, min_lat, data is in lat, lon, so use PlateCarree
                else:
                    llx, lly = self.thisarea.latlon_to_xy(bounding_lat - 12, -90)
                    urx, ury = self.thisarea.latlon_to_xy(bounding_lat - 12, -90 + 180)
                    ax.set_extent([llx, urx, lly, ury], dataprj)
            else:
                # Note that the '+1' below is a fudge to expand the area to account for the clipping
                # of the circular boundary
                ax.set_extent(
                    [-180, 180, -90, (-1) * bounding_lat + 1], ccrs.PlateCarree()
                )  # min lon, max_lon, max_lat, min_lat, data is in lat, lon, so use PlateCarree

            # Compute a circle in axes coordinates, which we can use as a boundary
            # for the map. We can pan/zoom as much as we like - the boundary will be
            # permanently circular.
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)

            ax.set_boundary(circle, transform=ax.transAxes)

        else:  # Normal area extent
            log.info("Normal area extent..")
            # Set the extent by a lower left lat,lon and a width and height in km
            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                log.info("specify_plot_area_by_lowerleft_corner..")
                llx, lly = self.thisarea.latlon_to_xy(
                    self.thisarea.llcorner_lat, self.thisarea.llcorner_lon
                )
                urx = llx + self.thisarea.width_km * 1000
                ury = lly + self.thisarea.height_km * 1000

                ax.set_extent([llx, urx, lly, ury], dataprj)

            # Set the extent by a centre lat,lon and a width and height in km
            elif self.thisarea.specify_by_centre:
                log.info("specify_by_centre..")
                cx, cy = self.thisarea.latlon_to_xy(
                    self.thisarea.centre_lat, self.thisarea.centre_lon
                )

                llx = cx - (self.thisarea.width_km * 1000) / 2.0
                lly = cy - (self.thisarea.height_km * 1000) / 2.0

                urx = llx + self.thisarea.width_km * 1000
                ury = lly + self.thisarea.height_km * 1000

                ax.set_extent([llx, urx, lly, ury], dataprj)

            # Set the extent by circular bounding latitude
            elif self.thisarea.specify_by_bounding_lat:
                log.info("Setting extent by bounding lat")
                if self.thisarea.hemisphere == "north":
                    if self.thisarea.lon_0 is not None and self.thisarea.lon_0 == 0.0:
                        # Note that the '-1' below is a fudge to expand the area to account for the
                        # clipping of the circular boundary
                        ax.set_extent(
                            [-180, 180, 90, self.thisarea.bounding_lat - 1],
                            ccrs.PlateCarree(),
                        )  # min lon, max_lon, max_lat, min_lat, data is in lat, lon,
                        # so use PlateCarree
                    else:
                        llx, lly = self.thisarea.latlon_to_xy(self.thisarea.bounding_lat - 12, -90)
                        urx, ury = self.thisarea.latlon_to_xy(
                            self.thisarea.bounding_lat - 12, -90 + 180
                        )
                        ax.set_extent([llx, urx, lly, ury], dataprj)
                else:
                    # Note that the '+1' below is a fudge to expand the area to account for the
                    # clipping of the circular boundary
                    ax.set_extent(
                        [-180, 180, -90, self.thisarea.bounding_lat + 1],
                        ccrs.PlateCarree(),
                    )  # min lon, max_lon, max_lat, min_lat, data is in lat, lon, so use PlateCarree

                # Make a circular border for the plot
                if self.thisarea.round:
                    # Compute a circle in axes coordinates, which we can use as a boundary
                    # for the map. We can pan/zoom as much as we like - the boundary will be
                    # permanently circular.
                    theta = np.linspace(0, 2 * np.pi, 100)
                    center, radius = [0.5, 0.5], 0.5
                    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                    circle = mpath.Path(verts * radius + center)

                    print("Setting round")

                    ax.set_boundary(circle, transform=ax.transAxes)

        return ax, dataprj, circle
