"""my_file_selector"""

import logging
import os
from glob import glob
from typing import List

from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.base.base_finder import BaseFinder

# pylint: disable=R0801
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals

log = logging.getLogger(__name__)


class FileFinder(BaseFinder):
    """class to find a list of files to process

    Args:
        BaseFinder (BaseFinder): base finder class

    In order to find files you can optionally use the following
    which are optionally set by the run_chain.py command line parameters

    Set by command line options:
        self.months  # list of months to find
        self.years   # list of years to find
    Set by config file settings:
        config["l1b_base_dir"]

    """

    def test_nc_file_in_area(self, file: str) -> str | None:
        """test if L1b file's global attributes indicate it is measuring over
        Greenland

        Args:
            file (str): path of CS2 L1b file

        Returns:
            bool : True if in the target area, False if not
        """
        try:
            with Dataset(file) as nc:
                if (
                    (
                        self.config["global"]["min_latitude"]
                        < nc.first_record_lat
                        < self.config["global"]["max_latitude"]
                    )
                    or (
                        self.config["global"]["min_latitude"]
                        < nc.last_record_lat
                        < self.config["global"]["max_latitude"]
                    )
                ) and (
                    (
                        self.config["global"]["min_longitude"]
                        < nc.first_record_lon
                        < self.config["global"]["max_longitude"]
                    )
                    or (
                        self.config["global"]["min_longitude"]
                        < nc.last_record_lon
                        < self.config["global"]["max_longitude"]
                    )
                ):
                    return file
                return None
        except IOError:
            return None

    # The base class is initialized with:
    # def __init__(self, log: logging.Logger | None = None, config: dict = {}):

    def find_files(self, flat_search=False) -> list[str]:
        # pylint: disable=too-many-statements
        """Search for L1b file according to pattern

        Args:
            flat_search (bool) : if True only search in l1b_base_dir, else use pattern
        Returns:
            (str): list of files

        """
        file_list: List[str] = []

        self.log.info("Finding files to process..")

        # Build a list of files to process based on
        # a) any configuration in self.config dictionary
        # b) any command line options set in self.months and self.years
        #   note these are lists of strings (so could contain more than
        #   one month or year) or None if not set

        if "month_file_dir" not in self.config["stage_3_file_finder"]:
            raise KeyError("month_file_dir missing from config")

        month_file_dir = self.config["stage_3_file_finder"]["month_file_dir"]

        file_search_string = "merge?(_NRT)_??????.nc"

        if flat_search:
            search_string = os.path.join(month_file_dir, file_search_string)
            files = glob(search_string)

            if len(files) > 0:
                file_list.extend(files)

        else:
            if len(self.years) < 1:
                raise ValueError("Empty year list. Use .add_years first.")

            if len(self.months) == 0:
                self.log.info("No months specified, using all months.")
                self.months: list[int] = list(range(1, 13))

            for year in self.years:
                self.log.info("Finding files for year: %d", year)
                for month in self.months:
                    self.log.info("Finding files for month: %d", month)

                    search_string = os.path.join(
                        month_file_dir, str(year), f"{month:02d}", file_search_string
                    )

                    files = glob(search_string)

                    if len(files) > 0:
                        file_list.extend(files)

        self.log.info("Total number of files found: %d", len(file_list))

        # some of the chain is optimised to process files from similar time periods more
        # efficiently, so sort them by time to try to encourage this
        if (
            "sort_files" in self.config["stage_3_file_finder"]
            and self.config["stage_3_file_finder"]["sort_files"]
        ):
            file_list = sorted(file_list)

        return file_list
