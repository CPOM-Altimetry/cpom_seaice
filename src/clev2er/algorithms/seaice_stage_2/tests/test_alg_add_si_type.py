"""pytest for algorithm
clev2er.algorithms.seaice.alg_add_si_type

Author: Ben Palmer
Date: 02 Jul 2024
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_si_type import Algorithm
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument


@pytest.fixture
def config() -> dict:
    """Pytest fixture for the config dictionary

    Returns:
        dict: config dictionary
    """
    # load config
    chain_config, _, _, _, _ = load_config_files("seaice")

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


merge_file_test = [(0), (1)]


@pytest.mark.parametrize("file_num", merge_file_test)
def test_add_si_type_sar(
    file_num,
    previous_steps: Dict,
    thisalg: Algorithm,
) -> None:
    """test alg_add_si_type.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that seaice_type is in shared_dict
    test that it is an array of floats
    test that it is equal in size to sat_lat
    test that all values in array are between 0 and 100
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing merge file:")

    # load SAR file
    l1b_merge_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "merge_modes").glob("*.nc")
    )[file_num]

    try:
        l1b = Dataset(l1b_merge_file)
        logger.info("Loaded %s", l1b_merge_file)
    except IOError:
        assert False, f"{l1b_merge_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert "seaice_type" in shared_dict, "SAR - Shared_dict does not contain 'seaice_type'"

    sit_dtype = shared_dict["seaice_type"].dtype
    assert (
        "float" in str(sit_dtype).lower()
    ), f"SAR - Dtype of 'seaice_type' is {sit_dtype}, not float"

    assert (
        shared_dict["seaice_type"].size == shared_dict["sat_lat"].size
    ), "SAR - 'seaice_type' is not the same length as 'sat_lat'"

    assert (
        sum((shared_dict["seaice_type"] <= 100.0) & (shared_dict["seaice_type"] >= 0.0)) > 0
    ), "SAR - 'seaice_type' contains incorrect values"
