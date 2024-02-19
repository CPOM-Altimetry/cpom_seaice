"""pytest for algorithm
    clev2er.algorithms.seaice.alg_flag_filter
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from numpy import ndarray

from clev2er.algorithms.seaice.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice.alg_flag_filters import Algorithm
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_flag_filters() -> None:
    """test alg_flag_filters.py
    Load an SAR and an SARIn files
    run Algorithm.process() on each
    test that the files return (True, "")
    check that indexes are within shared_dict
    check if all arrays have been filtered to the same size as sat_lon and sat_lat
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
    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # check if indices_flags array is in shared_dict
    assert "indices_flags" in shared_dict and isinstance(
        shared_dict["indices_flags"], ndarray
    ), "SAR - Flag indices not returned in shared_dict"

    # check if mcd_flags only contains 0s
    assert sum(shared_dict["mcd_flag"]) == 0, "SAR - Confidence flag contains non-zero values"

    # check if surface_type only contains 0s
    assert sum(shared_dict["surface_type"]) == 0, "SAR - Surface type flag contains non-zero values"

    # check if all fields from the file have been filtered to the same size
    assert all(
        len(shared_dict["sat_lat"]) == len(val)
        for key, val in shared_dict.items()
        if isinstance(val, ndarray) and "indices" not in key
    ), "SAR - Not all fields the same length"

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

    _, _ = ingest_cs2.process(l1b, shared_dict)  # ingest file
    _, _ = area_filter.process(l1b, shared_dict)  # filter area
    success, err_str = thisalg.process(l1b, shared_dict)  # filter by flags

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    # check if new indices array is in shared dict
    assert "indices_flags" in shared_dict and isinstance(
        shared_dict["indices_flags"], ndarray
    ), "SIN - Flag indices not returned in shared_dict"

    # check if mcd_flags only contains 0s
    assert sum(shared_dict["mcd_flag"]) == 0, "SIN - Confidence flag contains non-zero values"

    # check if surface_type only contains 0s
    assert sum(shared_dict["surface_type"]) == 0, "SIN - Surface type flag contains non-zero values"

    # check if all fields from the file have been filtered to the same size as sat_lat
    assert all(
        len(shared_dict["sat_lat"]) == len(val)
        for key, val in shared_dict.items()
        if isinstance(val, ndarray) and "indices" not in key
    ), "SIN - Not all fields the same length"
