"""clev2er.algorithms.seaice_stage_1.alg_sla_calculations.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Calculates sea level anomaly (SLA) using elevation and MSS. Removes SLA values greater than 20m
or less than -20m.

#Main initialization (init() function) steps/resources required

Get parameters from config file

#Main process() function steps

Find the raw SLA by subtracting elevation and mss
Remove any values outside of clipping range
Interpolate SLA between lead values using interp_sla
Filter out leads where the SLA is outside of the acceptable range
If leads in track have a mean SLA outside of limit, skip it

#Contribution to shared_dict

'raw_sea_level_anomaly'
'smoothed_sea_level_anomaly'
'lead_indx'
'indx_lead_sla_inside_range'

#Requires from shared_dict

'elevation'
'mss'
'lead_floe_class'
'sat_lat'
'sat_lon'

Author: Ben Palmer
Date: 12 Mar 2024
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm
from clev2er.utils.geo.interp_sea import interp_sea_regression


class Algorithm(BaseAlgorithm):
    """CLEV2ER Algorithm class

    contains:
            .log (Logger) : log instance that must be used for all logging, set by BaseAlgorithm
            .config (dict) : configuration dictionary, set by BaseAlgorithm

        functions that need completing:
            .init() : Algorithm initialization function (run once at start of chain)
            .process(l1b,shared_dict) : Algorithm processing function (run on every L1b file)
            .finalize() :   Algorithm finalization/closure function (run after all chain
                            processing completed)

        Inherits from BaseAlgorithm which handles interaction with the chain controller run_chain.py

    """

    def init(self) -> Tuple[bool, str]:
        """Algorithm initialization function

        Add steps in this function that are run once at the beginning of the chain
        (for example loading a DEM or Mask)

        Returns:
            (bool,str) : success or failure, error string

        Test for KeyError or OSError exceptions and raise them if found
        rather than just returning (False,"error description")

        Raises:
            KeyError : for keys not found in self.config
            OSError : for any file related errors

        Note:
        - retrieve required config data from self.config dict
        - log using self.log.info(), or self.log.error() or self.log.debug()

        """
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        # SLA related values
        self.clip_value = self.config["alg_sla_calculations"]["clip_value"]
        self.track_limit = self.config["alg_sla_calculations"]["track_limit"]
        self.sample_limit = self.config["alg_sla_calculations"]["sample_limit"]
        self.window_range = self.config["alg_sla_calculations"]["window_range"]
        self.distance_projection = self.config["alg_sla_calculations"]["distance_projection"]

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        """Main algorithm processing function, called for every L1b file

        Args:
            l1b (Dataset): input l1b file dataset (constant)
            shared_dict (dict): shared_dict data passed between algorithms. Use this dict
                                to pass algorithm results down the chain or read variables
                                set by other algorithms.

        Returns:
            Tuple : (success (bool), failure_reason (str))
            ie
            (False,'error string'), or (True,'')

        Note:
        - retrieve required config data from self.config dict (read-only)
        - retrieve data from other algorithms from shared_dict
        - add results,variables from this algorithm to shared_dict
        - log using self.log.info(), or self.log.error() or self.log.debug()

        """
        # pylint:disable=too-many-locals

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # \/    down the chain in the 'shared_dict' dict     \/
        # -------------------------------------------------------------------

        raw_sla = l1b["elevation"][:].data - shared_dict["mss"]

        # Remove values -clip<x<clip
        # From Andy:
        # Rejected anything more than 3m from MSS. A histogram
        # of SLA from specular echoes for cycle 013 shows almost
        # no data above +2m and below -1m.

        raw_sla[(raw_sla > self.clip_value) & (raw_sla < -self.clip_value)] = np.nan

        self.log.info("Number of NaNs in Raw SLA - %d", sum(np.isnan(raw_sla)))

        not_nan_sla = ~np.isnan(raw_sla)

        interp_lats = l1b["sat_lat"][:].data[not_nan_sla]
        interp_lons = l1b["sat_lon"][:].data[not_nan_sla]
        interp_sla = raw_sla[not_nan_sla]

        # find lead indices
        lead_indx = l1b["lead_floe_class"][:].data == 2
        interp_leads = lead_indx[not_nan_sla]

        if np.sum(lead_indx) == 0:
            self.log.info("No leads in file, unable to interpolate sea elevation")
            return (False, "SKIP_OK")

        interp_sla = interp_sea_regression(
            interp_lats,
            interp_lons,
            interp_sla,
            interp_leads,
            self.window_range * 1000,  # convert window_range from km to m
            self.distance_projection,
        )

        if interp_sla.size == 0:
            self.log.info("No SLA values found, skipping file")
            return (False, "SKIP_OK")

        self.log.info(
            "Interpolated sea elevation - Mean=%.3f Count=%d NaN=%d",
            np.nanmean(interp_sla),
            interp_sla.shape[0],
            sum(np.isnan(interp_sla)),
        )

        # find lead samples where sla is inside acceptable range
        indx_lead_sla_inside_range = np.isclose(interp_sla[interp_leads], 0, atol=self.sample_limit)

        self.log.info(
            "Leads with SLA outside of range - %d", np.sum(np.invert(indx_lead_sla_inside_range))
        )

        # remove leads with SLAs outside of acceptable values
        interp_sla[interp_leads][indx_lead_sla_inside_range] = np.nan

        # skip track if mean SLA of leads is outside of limit
        if not np.isclose(
            mean_sla := np.nanmean(interp_sla[interp_leads]), 0, atol=self.track_limit
        ):
            self.log.info("Mean SLA is outside of acceptable range - %f", mean_sla)
            self.log.info("Skipping file...")
            return (False, "SKIP_OK")

        self.log.info(
            "SLA - Mean=%.3f Std=%.3f Min=%.3f Max=%.3f Count=%d NaN=%d",
            np.nanmean(raw_sla),
            np.nanstd(raw_sla),
            np.nanmin(raw_sla),
            np.nanmax(raw_sla),
            raw_sla.shape[0],
            sum(np.isnan(raw_sla)),
        )

        smoothed_sla = np.zeros(raw_sla.size) * np.nan
        smoothed_sla[not_nan_sla] = interp_sla

        fmt_indx_lead_sla_inside_range = np.zeros(raw_sla.shape)
        fmt_indx_lead_sla_inside_range[not_nan_sla][interp_leads][indx_lead_sla_inside_range] = 1

        shared_dict["raw_sea_level_anomaly"] = raw_sla
        shared_dict["smoothed_sea_level_anomaly"] = smoothed_sla
        shared_dict["lead_indx"] = lead_indx
        shared_dict["indx_lead_sla_inside_range"] = fmt_indx_lead_sla_inside_range

        # -------------------------------------------------------------------
        # Returns (True,'') if success
        return (success, error_str)

    def finalize(self, stage: int = 0) -> None:
        """Algorithm finalization function - called after all processing completed

        Can be used to clean up/free resources initialized in the init() function

        Args:
            stage (int, optional):  this sets the stage when this function is called
                                    by the chain controller. Useful during multi-processing.
                                    Defaults to 0. Not normally used by Algorithms.
        """
        self.log.info(
            "Finalize algorithm %s called at stage %d filenum %d",
            self.alg_name,
            stage,
            self.filenum,
        )
        # ---------------------------------------------------------------------
        # Add finalization steps here \/
        # ---------------------------------------------------------------------

        # None

        # ---------------------------------------------------------------------
