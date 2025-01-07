"""pytest for algorithm
clev2er.algorithms.seaice_stage_3.alg_vol_calculations.py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_3.alg_vol_calculations import Algorithm
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


def test_vol_calculations(
    previous_steps: Dict,
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_add_region_mask.py

    Test plan:
    Load a merge file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that values added by vol_calculation are in shared_dict
    test that they are arrays
    test that they are either arrays of ints or arrays of floats depending on the variable
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

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # Algorithm tests
    for varname in [
        "volume_grid",
        "iceconc_grid",
        "thickness_grid",
        "frac_fyi_grid",
        "frac_myi_grid",
        "area_grid",
        "gaps",
        "number_in",
        "number_in_fyi",
        "number_in_myi",
    ]:
        assert varname in shared_dict, f"'{varname}' not in shared_dict."

        assert isinstance(
            shared_dict[varname], np.ndarray
        ), f"'{varname}' is {type(shared_dict[varname])}, not ndarray."

        mask_dtype = str(shared_dict["region_mask"].dtype)

        if varname in ["gaps", "number_in", "number_in_fyi", "number_in_myi"]:
            assert "int" in mask_dtype.lower(), f"Dtype of '{varname}' is {mask_dtype}, not int."
        else:
            assert (
                "float" in mask_dtype.lower()
            ), f"Dtype of '{varname}' is {mask_dtype}, not float."
