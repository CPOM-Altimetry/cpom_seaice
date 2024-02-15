"""pytest for algorithm
    clev2er.algorithms.seaice.alg_crop_waveform
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_crop_waveform import Algorithm
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_crop_waveform() -> None:
    """test alg_crop_waveform.py
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
        ingest_cs2 = IngestCS2(config, logger)  # no config used for this alg
        area_filter = AreaFilter(config, logger)
        flag_filter = FlagFilter(config, logger)
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

    _, _ = ingest_cs2.process(l1b, shared_dict)
    _, _ = area_filter.process(l1b, shared_dict)
    _, _ = flag_filter.process(l1b, shared_dict)

    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # check if all lats and lons are within the target area
    assert shared_dict["waveform"].shape[1] == 128, "SAR - waveform isn't in shape of Nx128"

    # check that bin_shift is within shared_dict
    assert "bin_shift" in shared_dict, "SAR - bin_shift not within shared dictionary"

    # ================================== SIN FILE TESTING ==========================================
    logger.info("Testing SIN file:")
    # load SARIn file
    l1b_sin_file = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sar").glob("*.nc")
    )[0]
    try:
        l1b = Dataset(l1b_sin_file)
        logger.info("Loaded %s", l1b_sin_file)
    except IOError:
        assert False, f"{l1b_sin_file} could not be read"

    shared_dict.clear()  # clear shared dictionary to reset it without reassignment

    _, _ = ingest_cs2.process(l1b, shared_dict)  # ingest file
    _, _ = area_filter.process(l1b, shared_dict)  # filter area
    _, _ = flag_filter.process(l1b, shared_dict)  # filter by flags

    success, err_str = thisalg.process(l1b, shared_dict)  # crop waveforms

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    # check if all lats and lons are within the target area
    assert shared_dict["waveform"].shape[1] == 128, "SIN - waveform isn't in shape of Nx128"

    # check that bin_shift is within shared_dict
    assert "bin_shift" in shared_dict, "SIN - bin_shift not within shared dictionary"
