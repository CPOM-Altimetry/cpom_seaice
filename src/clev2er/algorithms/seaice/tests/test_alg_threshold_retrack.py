"""pytest for algorithm
    clev2er.algorithms.seaice.alg_threshold_retrack
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm as CropWaveform
from clev2er.algorithms.seaice.alg_cs2_wave_discimination import (
    Algorithm as WaveDiscrimination,
)
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice.alg_pulse_peakiness import Algorithm as PulsePeakiness
from clev2er.algorithms.seaice.alg_smooth_waveform import Algorithm as SmoothWaveform
from clev2er.algorithms.seaice.alg_threshold_retrack import Algorithm
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
            "crop_waveform": CropWaveform(config, logger),
            "pulse_peakiness": PulsePeakiness(config, logger),
            "wave_discrim": WaveDiscrimination(config, logger),
            "smooth_waveform": SmoothWaveform(config, logger),
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


def test_retrack_floes_sar(
    previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_threshold_retrack.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # ================================== SAR FILE TESTING ==========================================
    logger.info("Testing SAR file:")

    # load SAR file
    l1b_sar_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sar").glob("*.nc")
    )[0]

    try:
        l1b = Dataset(l1b_sar_file)
        logger.info("Loaded %s", l1b_sar_file)
    except IOError:
        assert False, f"{l1b_sar_file} could not be read"

    shared_dict: Dict[str, Any] = {}

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert (
        "floe_retracking_points" in shared_dict
    ), "SAR - Shared_dict does not contain 'floe_retracking_points'"

    frp_dtype = shared_dict["floe_retracking_points"].dtype
    assert (
        "float" in str(frp_dtype).lower()
    ), f"SAR - Dtype of 'floe_retracking_points' is {frp_dtype}, not float"

    assert "idx_lew_lt_max" in shared_dict, "SAR - Shared_dict does not contain 'idx_lew_lt_max'"

    idx_lew_ftype = shared_dict["idx_lew_lt_max"].dtype
    assert (
        "int" in str(idx_lew_ftype).lower()
    ), f"SAR - Dtype of 'idx_lew_lt_max' is {idx_lew_ftype}, not int"


def test_retrack_floes_sin(
    previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_threshold_retrack.py for SIN waveforms

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
            logger.error("SIN - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert (
        "floe_retracking_points" in shared_dict
    ), "SIN - Shared_dict does not contain 'floe_retracking_points'"

    frp_dtype = shared_dict["floe_retracking_points"].dtype
    assert (
        "float" in str(frp_dtype).lower()
    ), f"SIN - Dtype of 'floe_retracking_points' is {frp_dtype}, not float"

    assert "idx_lew_lt_max" in shared_dict, "SIN - Shared_dict does not contain 'idx_lew_lt_max'"

    idx_lew_ftype = shared_dict["idx_lew_lt_max"].dtype
    assert (
        "int" in str(idx_lew_ftype).lower()
    ), f"SIN - Dtype of 'idx_lew_lt_max' is {idx_lew_ftype}, not int"
