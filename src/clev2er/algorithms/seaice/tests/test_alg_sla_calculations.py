"""pytest for algorithm
    clev2er.algorithms.seaice.alg_sla_calculations
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_add_mss import Algorithm as AddMss
from clev2er.algorithms.seaice.alg_add_si_conc import Algorithm as AddSIConc
from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm as CropWaveform
from clev2er.algorithms.seaice.alg_cs2_wave_discrimination import (
    Algorithm as WaveDiscrimination,
)
from clev2er.algorithms.seaice.alg_elev_calculations import Algorithm as ElevCalc
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_giles_retrack import Algorithm as GilesRetrack
from clev2er.algorithms.seaice.alg_ice_class import Algorithm as IceClass
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice.alg_pulse_peakiness import Algorithm as PulsePeakiness
from clev2er.algorithms.seaice.alg_sla_calculations import Algorithm
from clev2er.algorithms.seaice.alg_smooth_waveform import Algorithm as SmoothWaveform
from clev2er.algorithms.seaice.alg_threshold_retrack import (
    Algorithm as ThresholdRetrack,
)
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
            "ingest_cs2": IngestCS2(config, logger),
            "area_filter": AreaFilter(config, logger),
            "flag_filter": FlagFilter(config, logger),
            "add_si_conc": AddSIConc(config, logger),
            "crop_waveform": CropWaveform(config, logger),
            "pulse_peakiness": PulsePeakiness(config, logger),
            "wave_discrim": WaveDiscrimination(config, logger),
            "ice_class": IceClass(config, logger),
            "smooth_waveform": SmoothWaveform(config, logger),
            "threshold_retack": ThresholdRetrack(config, logger),
            "giles_retrack": GilesRetrack(config, logger),
            "elev_calculations": ElevCalc(config, logger),
            "add_mss": AddMss(config, logger),
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
def test_sla_calculations_sar(
    file_num, previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_sla_calculations.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that 'raw_sea_level_anomaly' and 'smoothed_sea_level_anomaly are in shared_dict, it is an
    array of floats, and values are all positive
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
            logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # Algorithm tests
    assert "raw_sea_level_anomaly" in shared_dict, "'raw_sea_level_anomaly' not in shared_dict."

    assert isinstance(
        shared_dict["raw_sea_level_anomaly"], np.ndarray
    ), f"'raw_sea_level_anomaly' is {type(shared_dict['raw_sea_level_anomaly'])}, not ndarray."

    elev_dtype = str(shared_dict["raw_sea_level_anomaly"].dtype)
    assert (
        "float" in elev_dtype.lower()
    ), f"Dtype of 'raw_sea_level_anomaly' is {elev_dtype}, not float."

    assert (
        "smoothed_sea_level_anomaly" in shared_dict
    ), "'smoothed_sea_level_anomaly' not in shared_dict."

    assert isinstance(shared_dict["smoothed_sea_level_anomaly"], np.ndarray), (
        f"'smoothed_sea_level_anomaly' is {type(shared_dict['smoothed_sea_level_anomaly'])},"
        "not ndarray."
    )

    elev_dtype = str(shared_dict["smoothed_sea_level_anomaly"].dtype)
    assert (
        "float" in elev_dtype.lower()
    ), f"Dtype of 'smoothed_sea_level_anomaly' is {elev_dtype}, not float."


def test_sla_calculations_sin(
    previous_steps: Dict, thisalg: Algorithm  # pylint: disable=redefined-outer-name
) -> None:
    """test alg_sla_calculations.py for SIN waveforms

    Test plan:
    Load a SARIn file
    run Algorithm.process() on each
    test that the files return (True, "")
    test that 'raw_sea_level_anomaly' and 'smoothed_sea_level_anomaly are in shared_dict, it is an
    array of floats, and values are all positive
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
    assert "raw_sea_level_anomaly" in shared_dict, "'raw_sea_level_anomaly' not in shared_dict."

    assert isinstance(
        shared_dict["raw_sea_level_anomaly"], np.ndarray
    ), f"'raw_sea_level_anomaly' is {type(shared_dict['raw_sea_level_anomaly'])}, not ndarray."

    elev_dtype = str(shared_dict["raw_sea_level_anomaly"].dtype)
    assert (
        "float" in elev_dtype.lower()
    ), f"Dtype of 'raw_sea_level_anomaly' is {elev_dtype}, not float."

    assert (
        "smoothed_sea_level_anomaly" in shared_dict
    ), "'smoothed_sea_level_anomaly' not in shared_dict."

    assert isinstance(shared_dict["smoothed_sea_level_anomaly"], np.ndarray), (
        f"'smoothed_sea_level_anomaly' is {type(shared_dict['smoothed_sea_level_anomaly'])},"
        "not ndarray."
    )

    elev_dtype = str(shared_dict["smoothed_sea_level_anomaly"].dtype)
    assert (
        "float" in elev_dtype.lower()
    ), f"Dtype of 'smoothed_sea_level_anomaly' is {elev_dtype}, not float."
