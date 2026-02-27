"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_merge_months.py
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_mss import Algorithm as AddMss
from clev2er.algorithms.seaice_stage_2.alg_add_si_type import Algorithm as AddSIType
from clev2er.algorithms.seaice_stage_2.alg_fbd_calculations import Algorithm as CalcFbd
from clev2er.algorithms.seaice_stage_2.alg_merge_months import Algorithm
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
    # load config
    chain_config, _, _, _, _ = load_config_files("seaice_stage_2")

    # Set to Sequential Processing
    chain_config["chain"]["use_multi_processing"] = False
    chain_config["alg_merge_modes"]["merge_file_dir"] = str(tmp_path)
    return chain_config


@pytest.fixture
def thisalg(config: Dict) -> Algorithm:
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


merge_file_test = [(0), (1)]


@pytest.mark.parametrize("file_num", merge_file_test)
def test_merge_months(
    file_num,
    tmpdir,
    previous_steps: Dict,
    thisalg: Algorithm,
) -> None:
    """test alg_grid_for_volume.py

    Test plan:
    process test file with previous steps
    run thisalg.process()
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
        f_time = datetime.fromtimestamp(np.min(l1b["measurement_time"]).astype(int)).strftime(
            "%Y%m"
        )
        merge_file_name = f"{f_time}_grids.nc"
    except IOError:
        assert False, f"{l1b_sar_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    thisalg.merge_file_dir = str(tmpdir)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # Algorithm tests
    month_output_file = os.path.join(str(tmpdir), merge_file_name)

    assert os.path.exists(month_output_file), f"Cannot find output merge file {month_output_file}"

    output_file = Dataset(month_output_file, "r")

    variables = {
        "packet_count"
        "block_number"
        "measurement_time"
        "valid"
        "sat_lat"
        "sat_lon"
        "thickness"
        "freeboard"
        "seaice_conc"
        "seaice_type"
    }

    assert (
        set(output_file.variables.keys()) == variables
    ), "Merge file does not contain all variables"
