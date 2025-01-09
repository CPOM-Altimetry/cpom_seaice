"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_add_ocean_frac
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_ocean_frac import Algorithm
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
        chain_previous_steps: Dict[str, Any] = {}
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


def test_add_ocean_frac_sar(
    previous_steps: Dict,
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_add_ocean_frac.py

    Test plan:
    Load a merge file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that 'ocean_frac' is in shared_dict, it is an array of floats
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== MERGE FILE TESTING ========================================
    logger.info("Testing merge file:")

    # load merge file
    l1b_merge_file = (
        base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "merge_modes" / "merge_060997.nc"
    )

    try:
        l1b = Dataset(l1b_merge_file)
        logger.info("Loaded %s", l1b_merge_file)
    except IOError:
        assert False, f"{l1b_merge_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error(" Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f" Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert "ocean_frac" in shared_dict, "'ocean_frac' not in shared_dict."

    assert isinstance(
        shared_dict["ocean_frac"], np.ndarray
    ), f"'ocean_frac' is {type(shared_dict['ocean_frac'])}, not ndarray."

    elev_dtype = str(shared_dict["ocean_frac"].dtype)
    assert "float" in elev_dtype.lower(), f"Dtype of 'ocean_frac' is {elev_dtype}, not float."
