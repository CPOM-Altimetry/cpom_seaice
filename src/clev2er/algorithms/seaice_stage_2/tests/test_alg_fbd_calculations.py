"""pytest for algorithm
clev2er.algorithms.seaice_stage_2.alg_fbd_calculations
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
from clev2er.algorithms.seaice_stage_2.alg_fbd_calculations import Algorithm
from clev2er.algorithms.seaice_stage_2.alg_sla_calculations import Algorithm as CalcSLA
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
            "warren_snow_means": WarrenSnowMeans(config, logger),
            "sla_calculations": CalcSLA(config, logger),
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


def test_fbd_calculations(
    previous_steps: Dict,  # pylint: disable=redefined-outer-name
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_fbd_calculations.py

    Test plan:
    Load a merge file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that 'freeboard' and 'freeboard_corr' are in shared_dict and they are arrays of floats
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing merge file:")

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
    assert "freeboard" in shared_dict, "'freeboard' not in shared_dict."

    assert isinstance(
        shared_dict["freeboard"], np.ndarray
    ), f"'freeboard' is {type(shared_dict['freeboard'])}, not ndarray."

    elev_dtype = str(shared_dict["freeboard"].dtype)
    assert "float" in elev_dtype.lower(), f"Dtype of 'freeboard' is {elev_dtype}, not float."

    assert "freeboard_corr" in shared_dict, "'freeboard_corr' not in shared_dict."

    assert isinstance(
        shared_dict["freeboard_corr"], np.ndarray
    ), f"'freeboard_corr' is {type(shared_dict['freeboard_corr'])}, not ndarray."

    elev_dtype = str(shared_dict["freeboard_corr"].dtype)
    assert "float" in elev_dtype.lower(), f"Dtype of 'freeboard_corr' is {elev_dtype}, not float."
