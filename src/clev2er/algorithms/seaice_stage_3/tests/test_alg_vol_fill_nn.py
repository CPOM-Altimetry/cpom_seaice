"""pytest for algorithm
clev2er.algorithms.seaice_stage_3.alg_vol_totals.py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_3.alg_vol_calculations import Algorithm as CalcVol
from clev2er.algorithms.seaice_stage_3.alg_vol_fill_nn import Algorithm
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
    chain_config, _, _, _, _ = load_config_files("seaice_stage_3")

    # Set to Sequential Processing
    chain_config["chain"]["use_multi_processing"] = False

    return chain_config


@pytest.fixture
def previous_steps(
    config: Dict,  # pylint: disable=unused-argument
) -> Dict:
    """Pytest fixture for generating the previous steps needed to test the algorithm

    Args:
        config (dict): Config fixture

    Returns:
        Dict: Dictionary of previous steps
    """
    ## Initialise the previous chain steps (needed to test current step properly)
    try:
        chain_previous_steps: Dict[str, Any] = {
            "vol_calculations": CalcVol(config, logger),
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
        this_algo = Algorithm(config, logger)
    except KeyError as exc:
        raise RuntimeError(f"Could not initialize algorithm {exc}") from exc

    return this_algo


def test_vol_total(
    previous_steps: Dict,
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_vol_total.py

    Test plan:
    Load a grid file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that volume_grid has been changed using the nn algorithm
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ==================================  FILE TESTING ==========================================
    logger.info("Testing  file:")

    # load  file
    grid_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "grid_files").glob("*.nc")
    )

    try:
        l1b = Dataset(grid_file)
        logger.info("Loaded %s", grid_file)
    except IOError:
        assert False, f"{grid_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    grid_before_fill = shared_dict["volume_grid"]

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # Algorithm tests
    grid_after_fill = shared_dict["volume_grid"]

    # Doesn't always mean that the algorithm didn't work, can improve with:
    # - Finding a test file that definitely had empty values to fill
    # - Finding a way of testing if a grid has places that are empty and can be filled using NNs
    assert not np.array_equal(grid_before_fill, grid_after_fill), "Algorithm had no effect on grid"
