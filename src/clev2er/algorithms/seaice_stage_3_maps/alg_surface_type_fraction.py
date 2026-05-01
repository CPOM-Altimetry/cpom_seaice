"""clev2er.algorithms.seaice.alg_surface_type_fraction.py


Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Grids points using latitude and longitude by finding all points within a set radius of each
grid cell. Uses the mean of found points.
Uses a grid defined by an external file.

#Main initialization (init() function) steps/resources required

Get config parameters

#Main process() function steps

make an array of unique packet ids
make empty arrays of lead, floe, ocean and unknown counts
for each unique packet id:
    get indices of samples with that packet id
    get number of floe, lead, ocean and unknown samples
    divide each by 20 (number of blocks in a packet)
    save where block 9 is
save data to shared dict


#Main finalize() function steps

None

#Contribution to shared_dict

floe_fraction
lead_fraction
ocean_fraction
unknown_fraction

#Requires from shared_dict

None

Author: Ben Palmer
Date: 20 Feb 2026
"""

from typing import Tuple

import numpy as np
from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    # pylint:disable=too-many-instance-attributes
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
        # pylint:disable=pointless-string-statement
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ 
        Get config parameters
        Check that output directory exists
        """

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint: disable=pointless-string-statement
        # pylint: disable=too-many-statements
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

        # This step is required to support multi-processing. Do not modify
        success, error_str = self.process_setup(l1b)
        if not success:
            return (False, error_str)

        # -------------------------------------------------------------------
        # Perform the algorithm processing, store results that need to be passed
        # /    down the chain in the 'shared_dict' dict     /
        # -------------------------------------------------------------------

        """ 
        make an array of unique packet ids
        make empty arrays of lead, floe, ocean and unknown counts
        for each unique packet id:
            get indices of samples with that packet id
            get number of floe, lead, ocean and unknown samples
            divide each by 20 (number of blocks in a packet)
            save where block 9 is 
        save data to shared dict
        """

        packet_count = l1b["packet_count"][:].data.flatten()
        block_number = l1b["block_number"][:].data.flatten()
        surface_type = l1b["surface_type"][:].data.flatten()
        concentration = l1b["seaice_conc"][:].data.flatten()
        valid = l1b["elev_valid"][:].data.flatten().astype(np.bool_) & (concentration > 15.0)

        changes = np.concatenate(([True], packet_count[1:] != packet_count[:-1]))
        unique_packet_id = np.cumsum(changes) - 1
        self.log.info("Found %d packets", np.max(unique_packet_id))

        block_nine = block_number == 9
        floe_samples = (surface_type == 2) & valid
        lead_samples = (surface_type == 3) & valid
        ocean_samples = (surface_type == 1) & valid
        unknown_samples = (surface_type == 0) & ~valid

        n_floe = np.full_like(packet_count, np.nan, dtype=np.float64)
        n_lead = np.full_like(packet_count, np.nan, dtype=np.float64)
        n_ocean = np.full_like(packet_count, np.nan, dtype=np.float64)
        n_unknown = np.full_like(packet_count, np.nan, dtype=np.float64)

        # Sum per-packet counts for each surface type
        n_floe_counts = np.bincount(unique_packet_id, weights=floe_samples.astype(float))
        n_lead_counts = np.bincount(unique_packet_id, weights=lead_samples.astype(float))
        n_ocean_counts = np.bincount(unique_packet_id, weights=ocean_samples.astype(float))
        n_unknown_counts = np.bincount(unique_packet_id, weights=unknown_samples.astype(float))

        # Assign back only to block_nine positions
        block_nine_mask = block_nine  # boolean array
        n_floe[block_nine_mask] = n_floe_counts[unique_packet_id[block_nine_mask]]
        n_lead[block_nine_mask] = n_lead_counts[unique_packet_id[block_nine_mask]]
        n_ocean[block_nine_mask] = n_ocean_counts[unique_packet_id[block_nine_mask]]
        n_unknown[block_nine_mask] = n_unknown_counts[unique_packet_id[block_nine_mask]]

        # work out fractions and save to shared_dict
        floe_frac = n_floe / 20
        lead_frac = n_lead / 20
        ocean_frac = n_ocean / 20
        unknown_frac = n_unknown / 20

        self.log.info("Mean floe frac - %0.2f", np.nanmean(floe_frac))
        self.log.info("Mean leads frac - %0.2f", np.nanmean(lead_frac))
        self.log.info("Mean ocean frac - %0.2f", np.nanmean(ocean_frac))
        self.log.info("Mean unknown frac - %0.2f", np.nanmean(unknown_frac))

        shared_dict["floe_fraction"] = floe_frac
        shared_dict["lead_fraction"] = lead_frac
        shared_dict["ocean_fraction"] = ocean_frac
        shared_dict["unknown_fraction"] = unknown_frac

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
        # pylint:disable=pointless-string-statement

        # ---------------------------------------------------------------------
        # Add finalization steps here /
        # ---------------------------------------------------------------------

        """ None """

        # ---------------------------------------------------------------------
