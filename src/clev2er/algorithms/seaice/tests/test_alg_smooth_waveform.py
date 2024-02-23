"""pytest for algorithm
    clev2er.algorithms.seaice.alg_smooth_waveform
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm as CropWaveform
from clev2er.algorithms.seaice.alg_cs2_wave_discimination import (
    Algorithm as WaveDiscrimination,
)
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice.alg_pulse_peakiness import Algorithm as PulsePeakiness
from clev2er.algorithms.seaice.alg_smooth_waveform import Algorithm
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_smooth_waveform() -> None:
    """test alg_smooth_waveform.py
    Load an SAR and an SARIn files
    run Algorithm.process() on each
    test that the files return (True, "")
    Check that shape of input waveform is the same as shape of output waveforms
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # load config
    config, _, _, _, _ = load_config_files("seaice")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    ## Initialise the previous chain steps (needed to test current step properly)
    try:
        previous_steps = {
            "ingest_cs2": IngestCS2(config, logger),  # no config used for this alg
            "area_filter": AreaFilter(config, logger),
            "flag_filter": FlagFilter(config, logger),
            "crop_waveform": CropWaveform(config, logger),
            "pulse_peakiness": PulsePeakiness(config, logger),
            "wave_discrim": WaveDiscrimination(config, logger),
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
            assert success, f"SAR - Algorithm failed due to: {err_str}"

    success, err_str = thisalg.process(l1b, shared_dict)

    # Algorithm tests

    assert (out_s := shared_dict["waveform_smooth"].shape) == (
        in_s := shared_dict["waveform"][shared_dict["diffuse_index"]].shape
    ), f"SAR - Input shape is different from output shape. In={in_s} Out={out_s}"

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

    shared_dict.clear()  # clear shared dictionary to reset it without reassignment

    for title, step in previous_steps.items():
        success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
        if not success:
            logger.error("SIN - Error with previous step: %s\n%s", title, err_str)
            assert success, f"SIN - Algorithm failed due to: {err_str}"

    success, err_str = thisalg.process(l1b, shared_dict)

    # Algorithm tests
    assert (out_s := shared_dict["waveform_smooth"].shape) == (
        in_s := shared_dict["waveform"][shared_dict["diffuse_index"]].shape
    ), f"SIN - Input shape is different from output shape. In={in_s} Out={out_s}"
