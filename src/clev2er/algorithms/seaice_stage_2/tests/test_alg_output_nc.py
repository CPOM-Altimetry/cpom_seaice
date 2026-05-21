"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_output_nc.py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_mss import Algorithm as AddMss
from clev2er.algorithms.seaice_stage_2.alg_add_si_type import Algorithm as AddSIType
from clev2er.algorithms.seaice_stage_2.alg_fbd_calculations import Algorithm as CalcFbd
from clev2er.algorithms.seaice_stage_2.alg_output_nc import Algorithm
from clev2er.algorithms.seaice_stage_2.alg_sla_calculations import Algorithm as CalcSLA
from clev2er.algorithms.seaice_stage_2.alg_thk_calculations import Algorithm as CalcThk
from clev2er.algorithms.seaice_stage_2.alg_warren_snow_means import (
    Algorithm as WarrenSnowMeans,
)
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)

# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals


@pytest.fixture
def config(tmp_path) -> dict:
    """Pytest fixture for the config dictionary

    Returns:
        dict: config dictionary
    """
    chain_config, _, _, _, _ = load_config_files("seaice_stage_2")

    chain_config["chain"]["use_multi_processing"] = False
    chain_config["alg_output_nc"]["output_dir"] = str(tmp_path)
    return chain_config


@pytest.fixture
def thisalg(config: Dict) -> Algorithm:
    """Pytest fixture for the main algorithm being tested in this file

    Args:
        config (dict): Pytest fixture for the chain config

    Returns:
        Algorithm: Returns the algorithm instance
    """
    try:
        this_algo = Algorithm(config, logger)
    except KeyError as exc:
        raise RuntimeError(f"Could not initialize algorithm {exc}") from exc

    return this_algo


@pytest.fixture
def previous_steps(config: Dict) -> Dict:
    """Pytest fixture for generating the previous steps needed to test the algorithm

    Args:
        config (dict): Config fixture

    Returns:
        Dict: Dictionary of previous steps
    """
    try:
        chain_previous_steps = {
            "add_mss": AddMss(config, logger),
            "add_si_type": AddSIType(config, logger),
            "sla_calculations": CalcSLA(config, logger),
            "warren_snow_means": WarrenSnowMeans(config, logger),
            "fbd_calculations": CalcFbd(config, logger),
            "thk_calculations": CalcThk(config, logger),
        }
    except KeyError as exc:
        raise RuntimeError(f"Could not initialize previous steps in chain {exc}") from exc

    return chain_previous_steps


output_nc_file_test = [(0), (1)]


@pytest.mark.parametrize("file_num", output_nc_file_test)
def test_output_nc(
    file_num,
    tmp_path,
    previous_steps: Dict,
    thisalg: Algorithm,
) -> None:
    """Test alg_output_nc.py

    Test plan:
        - Process a test L1b file through the preceding chain steps to populate shared_dict
        - Run thisalg.process() on the same file
        - Assert that process() returns (True, "")
        - Assert that the output .nc file is created under the expected year/month subdirectory
        - Assert that the output file contains all expected variables
        - Assert that the output file contains the orbit_number global attribute
        - Assert that all variables have the same length as measurement_time
        - Assert that measurement_time values are sorted in ascending order
    """
    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    logger.info("Testing output nc files:")

    l1b_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "merge_modes").glob("*.nc")
    )[file_num]

    try:
        l1b = Dataset(l1b_file)
        logger.info("Loaded %s", l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    # Point the algorithm at the tmp directory so output files are created there
    thisalg.output_dir = str(tmp_path)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # -----------------------------------------------------------------------
    # Locate the output file
    # -----------------------------------------------------------------------
    # The algorithm writes files to <output_dir>/<YYYY>/<MM>/ — find it by
    # walking the tmp directory rather than reconstructing the exact filename.
    nc_files = list(tmp_path.rglob("cs2_arc_*.nc"))
    assert (
        len(nc_files) == 1
    ), f"Expected exactly 1 output .nc file, found {len(nc_files)}: {nc_files}"

    output_file_path = nc_files[0]
    assert output_file_path.exists(), f"Output file not found: {output_file_path}"

    # -----------------------------------------------------------------------
    # Validate the year/month subdirectory structure
    # -----------------------------------------------------------------------
    # Path should be <output_dir>/<YYYY>/<MM>/cs2_arc_*.nc
    parts = output_file_path.parts
    month_dir = parts[-2]  # e.g. "11"
    year_dir = parts[-3]  # e.g. "2020"
    assert (
        len(year_dir) == 4 and year_dir.isdigit()
    ), f"Expected a 4-digit year directory, got '{year_dir}'"
    assert (
        len(month_dir) == 2 and month_dir.isdigit()
    ), f"Expected a 2-digit month directory, got '{month_dir}'"

    # -----------------------------------------------------------------------
    # Validate contents of the output file
    # -----------------------------------------------------------------------
    with Dataset(output_file_path, "r") as output_nc:
        # Check all expected variables are present
        expected_variables = {
            "packet_count",
            "block_number",
            "measurement_time",
            "thk_valid",
            "elev_valid",
            "sat_lat",
            "sat_lon",
            "surface_type",
            "thickness",
            "freeboard",
            "seaice_conc",
            "seaice_type",
            "floe_chord_length",
            "snow_depth",
            "sea_level_anomaly",
        }
        assert expected_variables.issubset(set(output_nc.variables.keys())), (
            f"Output file is missing variables. "
            f"Missing: {expected_variables - set(output_nc.variables.keys())}"
        )

        # Check that global attributes are written
        assert hasattr(
            output_nc, "orbit_number"
        ), "Output file is missing the 'orbit_number' global attribute"
        assert hasattr(
            output_nc, "min_time"
        ), "Output file is missing the 'min_time' global attribute"
        assert hasattr(
            output_nc, "max_time"
        ), "Output file is missing the 'max_time' global attribute"

        # Check that all variables share the same length
        n_samples = len(output_nc.dimensions["n_samples"])
        for var_name in expected_variables:
            var_len = output_nc[var_name][:].shape[0]
            assert (
                var_len == n_samples
            ), f"Variable '{var_name}' has length {var_len}, expected {n_samples}"

        # Check that measurement_time is sorted in ascending order
        measurement_time = output_nc["measurement_time"][:].data
        assert np.all(
            measurement_time[:-1] <= measurement_time[1:]
        ), "measurement_time values in the output file are not sorted in ascending order"
