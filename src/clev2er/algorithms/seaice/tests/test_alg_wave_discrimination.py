"""pytest for algorithm
    clev2er.algorithms.seaice.alg_cs2_wave_discrimination.py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm as CropWaveform
from clev2er.algorithms.seaice.alg_cs2_wave_discrimination import Algorithm
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice.alg_pulse_peakiness import Algorithm as PulsePeakiness
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_cs2_wave_discrimination() -> None:
    """test alg_cs2_wave_discrimination.py
    Load an SAR and an SARIn files
    run Algorithm.process() on each
    test that the files return (True, "")
    check that the length of "waveform"'s second axis is equal to 128
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # load config
    config, _, _, _, _ = load_config_files("seaice")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    # Initialise the previous chain steps (needed to test current step properly)
    try:
        previous_steps = {
            "ingest_cs2": IngestCS2(config, logger),  # no config used for this alg
            "area_filter": AreaFilter(config, logger),
            "flag_filter": FlagFilter(config, logger),
            "crop_waveform": CropWaveform(config, logger),
            "pulse_peakiness": PulsePeakiness(config, logger),
        }
    except KeyError as exc:
        assert False, f"Could not initialize previous steps in chain {exc}"

    # Initialise the Algorithm
    try:
        thisalg = Algorithm(config, logger)  # no config used for this alg
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

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

    for key in ["specular_index", "diffuse_index"]:
        # check that specular_index, diffuse_index and lead_floe_class are within shared_dict
        assert key in shared_dict, f"SAR - {key} not within shared dictionary"

        # check that the above values are arrays of ints
        assert isinstance(shared_dict[key], np.ndarray)
        assert (
            "int" in str(shared_dict[key].dtype).lower()
        ), f"SAR - {key} is not an array of ints - {shared_dict[key].dtype}"

    # ================================== SIN FILE TESTING ========================================
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

    shared_dict.clear()  # clear shared dictionary to reset it without reassignment

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    for key in ["specular_index", "diffuse_index"]:
        # check that specular_index, diffuse_index and lead_floe_class are within shared_dict
        assert key in shared_dict, f"SIN - {key} not within shared dictionary"

        # check that the above values are arrays of ints
        assert isinstance(shared_dict[key], np.ndarray)
        assert "int" in str(
            shared_dict[key].dtype
        ), f"SIN - {key} is not an array of ints - {shared_dict[key].dtype}"
