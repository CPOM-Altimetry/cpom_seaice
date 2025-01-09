"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_warren_snow_means
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_2.alg_add_si_type import Algorithm as AddIceType
from clev2er.algorithms.seaice_stage_2.alg_warren_snow_means import Algorithm
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
            "add_ice_type": AddIceType(config, logger),
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


def test_warren_snow_means(
    previous_steps: Dict,  # pylint: disable=redefined-outer-name
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_fbd_calculations.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that 'freeboard' is in shared_dict, it is an array of floats, and
        values are all positive
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing SAR file:")

    # load SAR file
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
            logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # Algorithm tests
    # tests for depth
    assert "snow_depth" in shared_dict, "'snow_depth' not in shared_dict."

    assert isinstance(
        shared_dict["snow_depth"], np.ndarray
    ), f"'snow_depth' is {type(shared_dict['snow_depth'])}, not ndarray."

    depth_dtype = str(shared_dict["snow_depth"].dtype)
    assert "float" in depth_dtype.lower(), f"Dtype of 'snow_depth' is {depth_dtype}, not float."

    # tests of density
    assert "snow_density" in shared_dict, "'snow_density' not in shared_dict."

    assert isinstance(
        shared_dict["snow_density"], np.ndarray
    ), f"'snow_density' is {type(shared_dict['snow_density'])}, not ndarray."

    density_dtype = str(shared_dict["snow_density"].dtype)
    assert (
        "float" in density_dtype.lower()
    ), f"Dtype of 'snow_density' is {density_dtype}, not float."
