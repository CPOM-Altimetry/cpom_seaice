"""pytest for algorithm
    clev2er.algorithms.seaice.alg_ice_class

Author: Ben Palmer
Date: 08 Mar 2024
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_add_si_conc import Algorithm as AddSIConc
from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm as CropWaveform
from clev2er.algorithms.seaice.alg_cs2_wave_discrimination import (
    Algorithm as WaveDiscrimination,
)
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_ice_class import Algorithm
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice.alg_pulse_peakiness import Algorithm as PulsePeakiness
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


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
        chain_previous_steps = {
            "ingest_cs2": IngestCS2(config, logger),  # no config used for this alg
            "area_filter": AreaFilter(config, logger),
            "flag_filter": FlagFilter(config, logger),
            "add_si_conc": AddSIConc(config, logger),
            "crop_waveform": CropWaveform(config, logger),
            "pulse_peakiness": PulsePeakiness(config, logger),
            "wave_discrim": WaveDiscrimination(config, logger),
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


sar_file_test = [(0), (1)]


@pytest.mark.parametrize("file_num", sar_file_test)
def test_retrack_leads_sar(
    file_num, previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_ice_class.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that every sample has a class value
    test that all class values are within range of 0-3
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing SAR file:")

    # load SAR file
    l1b_sar_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sar").glob("*.nc")
    )[file_num]

    try:
        l1b = Dataset(l1b_sar_file)
        logger.info("Loaded %s", l1b_sar_file)
    except IOError:
        assert False, f"{l1b_sar_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert "lead_floe_class" in shared_dict, "Shared_dict does not contain 'lead_floe_class'"

    lfc_dtype = shared_dict["lead_floe_class"].dtype
    assert (
        "int" in str(lfc_dtype).lower()
    ), f"Dtype of 'lead_retracking_points' is {lfc_dtype}, not int"

    assert (
        shared_dict["lead_floe_class"].size == shared_dict["sat_lat"].size
    ), "Class value missing from sample(s)"

    assert (
        sum((shared_dict["lead_floe_class"] < 0) & (shared_dict["lead_floe_class"] > 3)) == 0
    ), "Class array contains invalid values"


def test_ice_class_sin(
    previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_ice_class.py for SIN waveforms

    Test plan:
    Load a SARIn file
    run Algorithm.process() on each
    test that the files return (True, "")
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SIN FILE TESTING ==========================================

    logger.info("Testing SIN file:")
    # load SARIn file
    l1b_sin_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sin").glob("*.nc")
    )[0]
    try:
        l1b = Dataset(l1b_sin_file)
        logger.info("Loaded %s", l1b_sin_file)
    except IOError:
        assert False, f"{l1b_sin_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert "lead_floe_class" in shared_dict, "Shared_dict does not contain 'lead_floe_class'"

    lfc_dtype = shared_dict["lead_floe_class"].dtype
    assert (
        "int" in str(lfc_dtype).lower()
    ), f"Dtype of 'lead_retracking_points' is {lfc_dtype}, not int"

    assert (
        shared_dict["lead_floe_class"].size == shared_dict["sat_lat"].size
    ), "Class value missing from sample(s)"

    assert (
        sum((shared_dict["lead_floe_class"] < 0) | (shared_dict["lead_floe_class"] > 3)) == 0
    ), "Class array contains invalid values"
