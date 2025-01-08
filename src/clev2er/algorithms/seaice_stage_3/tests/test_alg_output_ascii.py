"""pytest for algorithm
clev2er.algorithms.seaice_stage_3.alg_output_ascii.py
"""

import fnmatch
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_3.alg_output_ascii import Algorithm
from clev2er.algorithms.seaice_stage_3.alg_vol_calculations import Algorithm as CalcVol
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

    # Set any testing config here

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
        chain_previous_steps: Dict[str, Any] = {"volume_calculations": CalcVol(config, logger)}
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


def test_output_ascii(
    previous_steps: Dict,
    thisalg: Algorithm,  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_output_ascii.py

    Test plan:
    Load a test file
    run Algorithm.process() on each
    test that the files return (True, "")


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

    temp_dir = TemporaryDirectory()  # pylint:disable=consider-using-with

    thisalg.output_directory = Path(temp_dir.name)

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"Algorithm failed due to: {err_str}"

    # Algorithm tests
    output_files = os.listdir(temp_dir.name)

    assert len(fnmatch.filter(output_files, "*.vol")) >= 1, ".vol file wasn't created"
    assert len(fnmatch.filter(output_files, "*.thk")) >= 1, ".thk file wasn't created"
    assert len(fnmatch.filter(output_files, "*.gaps")) >= 1, ".gaps file wasn't created"
    assert len(fnmatch.filter(output_files, "*.area")) >= 1, ".area file wasn't created"
    assert len(fnmatch.filter(output_files, "*.conc")) >= 1, ".conc file wasn't created"

    temp_dir.cleanup()
