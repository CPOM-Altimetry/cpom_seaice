"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_grid_for_volume
"""
# pylint:disable=import-error
# pylint:disable=no-name-in-module

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_mss import Algorithm as AddMss
from clev2er.algorithms.seaice_stage_2.alg_add_si_type import Algorithm as AddSIType
from clev2er.algorithms.seaice_stage_2.alg_fbd_calculations import Algorithm as CalcFbd
from clev2er.algorithms.seaice_stage_2.alg_grid_for_volume import Algorithm
from clev2er.algorithms.seaice_stage_2.alg_sla_calculations import Algorithm as CalcSLA
from clev2er.algorithms.seaice_stage_2.alg_thk_calculations import Algorithm as CalcThk
from clev2er.algorithms.seaice_stage_2.alg_warren_snow_means import (
    Algorithm as WarrenSnowMeans,
)
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)
# pylint: disable=redefined-outer-name


@pytest.fixture
def config() -> dict:
    """Pytest fixture for the config dictionary

    Returns:
        dict: config dictionary
    """
    # load config
    chain_config, _, _, _, _ = load_config_files("seaice_stage_2")

    # Set to Sequential Processing
    chain_config["chain"]["use_multi_processing"] = False

    return chain_config


@pytest.fixture
def previous_steps(
    config: Dict,  # pylint: disable=redefined-outer-name
) -> Dict:
    """Pytest fixture for generating the previous steps needed to test the algorithm

    Args:
        config (dict): Config fixture

    Returns:
        Dict: Dictionary of previous steps
    """
    ## Initialise the previous chain steps (needed to test current step properly)
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


@pytest.fixture
def thisalg(config: Dict) -> Algorithm:  # pylint: disable=redefined-outer-name
    """Pytest fixture for the main algorithm being tested in this file

    Args:
        config (dict): Pytest fixture for the chain config

    Returns:
        Any: Returns the algorithm
    """
    # Initialise the Algorithm
    try:
        this_algo = Algorithm(config, logger)  # no config used for this alg
    except KeyError as exc:
        raise RuntimeError(f"Could not initialize algorithm {exc}") from exc

    return this_algo


merge_file_test = [(0), (1)]


@pytest.mark.parametrize("file_num", merge_file_test)
def test_grid_for_volume(
    file_num,
    previous_steps: Dict,
    thisalg: Algorithm,
) -> None:
    """test alg_grid_for_volume.py

    Test plan:
    Load a merge file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that output file is created
    test that output file contains correct variables
    """
    # pylint:disable=too-many-locals
    # pylint:disable=consider-using-with

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing merge files:")

    # load SAR file
    l1b_sar_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "merge_modes").glob("*.nc")
    )[file_num]

    try:
        l1b = Dataset(l1b_sar_file)
        logger.info("Loaded %s", l1b_sar_file)
        f_time = datetime.fromtimestamp(np.min(l1b["measurement_time"])).strftime("%Y%M")
        grid_file_name = f"{f_time}_grids.nc"
    except IOError:
        assert False, f"{l1b_sar_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    temp_dir = tempfile.TemporaryDirectory()
    thisalg.grid_directory = temp_dir.name

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # Algorithm tests
    grid_output_file = os.path.join(temp_dir.name, grid_file_name)

    assert os.path.exists(grid_output_file), f"Cannot find output grid file {grid_output_file}"

    output_file = Dataset(grid_output_file, "r")

    variables = {
        "thickness",
        "thickness_fyi",
        "thickness_myi",
        "iceconc",
        "number_in",
        "number_in_fyi",
        "number_in_myi",
    }

    assert (
        set(output_file.variables.keys()) == variables
    ), "Grid file does not contain all variables"

    temp_dir.cleanup()
