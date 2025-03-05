"""clev2er.algorithms.seaice.alg_output_ascii.py

Algorithm class module, used to implement a single chain algorithm

#Description of this Algorithm's purpose

Outputs the gridded results of the processing chain in ASCII file formats, the same format as the
original chain. Outputs a file for each of volume, gaps, area, thickness and concentration.

#Main initialization (init() function) steps/resources required

Get config parameters
Check that output directory exists

#Main process() function steps

Get year and month of input file
If it does not exist, create folder for year and for outputs
Open output file as text files
Loop through lats and lons
Write line containing relevant information

#Main finalize() function steps

None

#Contribution to shared_dict

None

#Requires from shared_dict

volume_grid
thickness_grid
gaps
area_grid
iceconc_grid

Author: Ben Palmer
Date: 08 Jan 2025
"""

import os
from pathlib import Path
from typing import Tuple

from codetiming import Timer
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_alg import BaseAlgorithm


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
        # pylint:disable=pointless-string-statement
        self.alg_name = __name__
        self.log.info("Algorithm %s initializing", self.alg_name)

        # --- Add your initialization steps below here ---

        """ 
        Get config parameters
        Check that output directory exists
        """

        self.nlats = self.config["shared"]["grid_nlats"]
        self.nlons = self.config["shared"]["grid_nlons"]
        self.output_directory = Path(self.config["alg_output_ascii"]["output_directory"])

        if not self.output_directory.exists():
            os.makedirs(self.output_directory)

        # --- End of initialization steps ---

        return (True, "")

    @Timer(name=__name__, text="", logger=None)
    def process(self, l1b: Dataset, shared_dict: dict) -> Tuple[bool, str]:
        # pylint: disable=too-many-locals
        # pylint: disable=unpacking-non-sequence
        # pylint:disable=pointless-string-statement
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
        Get year and month of input file
        If it does not exist, create folder for year and for outputs
        Open output file as text files
        Loop through lats and lons
        Write line containing relevant information
        """

        # check if year folder exists
        year_folder = self.output_directory / l1b.year
        if not year_folder.exists():
            os.makedirs(year_folder)

        filename = f"{l1b.year:04d}_{l1b.month:02d}"

        vol_file = year_folder / (filename + ".vol")
        thickness_file = year_folder / (filename + ".thk")
        gaps_file = year_folder / (filename + ".gaps")
        area_file = year_folder / (filename + ".area")
        conc_file = year_folder / (filename + ".conc")

        with (
            open(vol_file, mode="+w", encoding="ascii") as vol_fp,
            open(thickness_file, mode="+w", encoding="ascii") as thick_fp,
            open(gaps_file, mode="+w", encoding="ascii") as gaps_fp,
            open(area_file, mode="+w", encoding="ascii") as area_fp,
            open(conc_file, mode="+w", encoding="ascii") as conc_fp,
        ):
            for ilat in range(0, self.nlats):
                lat = 40 + (ilat * 0.1)
                for ilon in range(0, self.nlons):
                    lon = -180 + (ilon * 0.5)
                    vol_fp.write(
                        f"{ilat: 4d}{ilon: 4d}{lat: 10.4f}{lon: 10.4f}"
                        f"{shared_dict['volume_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['frac_fyi_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['frac_myi_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['number_in'][ilat, ilon]: 6d}"
                        "\n"
                    )
                    gaps_fp.write(
                        f"{ilat: 4d}{ilon: 4d}{lat: 10.4f}{lon: 10.4f}"
                        f"{shared_dict['gaps'][ilat, ilon]: 4d}"
                        "\n"
                    )
                    area_fp.write(
                        f"{ilat: 4d}{ilon: 4d}{lat: 10.4f}{lon: 10.4f}"
                        f"{shared_dict['area_grid'][ilat, ilon]: 10.2f}"
                        f"{shared_dict['frac_fyi_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['frac_myi_grid'][ilat, ilon]: 10.4f}"
                    )
                    thick_fp.write(
                        f"{ilat: 4d}{ilon: 4d}{lat: 10.4f}{lon: 10.4f}"
                        f"{shared_dict['thickness_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['number_in'][ilat, ilon]: 6d}"
                    )
                    conc_fp.write(
                        f"{ilat: 4d}{ilon: 4d}{lat: 10.4f}{lon: 10.4f}"
                        f"{shared_dict['iceconc_grid'][ilat, ilon]: 10.4f}"
                        f"{shared_dict['number_in'][ilat, ilon]: 6d}"
                    )

        self.log.info("Saved volume to %s", vol_file)
        self.log.info("Saved thickness to %s", thickness_file)
        self.log.info("Saved gaps to %s", gaps_file)
        self.log.info("Saved area to %s", area_file)
        self.log.info("Saved concentration to %s", conc_file)

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
