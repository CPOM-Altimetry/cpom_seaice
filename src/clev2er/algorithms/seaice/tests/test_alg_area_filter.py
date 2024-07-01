"""pytest for algorithm
clev2er.algorithms.seaice.alg_area_filter
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from numpy import ndarray

from clev2er.algorithms.seaice.alg_area_filter import Algorithm
from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_area_filter() -> None:
    """test alg_ingest_cs2.py
    Load an SAR and an SARIn files
    run Algorithm.process() on each
    test that the files return (True, "")
    check if all of sat_lat and sat_lon are within constraints
    check if all arrays have been filtered to the same size as sat_lon and sat_lat
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # load config
    config, _, _, _, _ = load_config_files("seaice")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    # Initialise the alg_ingest_cs2 algorithm
    try:
        ingest_cs2 = IngestCS2(config, logger)  # no config used for this alg
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

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
    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"

    # check if all lats and lons are within the target area
    assert all(
        (
            config["shared"]["min_longitude"] <= lon <= config["shared"]["max_longitude"]
            and config["shared"]["min_latitude"] <= lat <= config["shared"]["max_latitude"]
        )
        for lon, lat in zip(shared_dict["sat_lon"], shared_dict["sat_lat"])
    ), "SAR - Contains lat-lon values outside of area filter range"

    # check if all fields from the file have been filtered to the same size as sat_lat
    assert all(
        len(shared_dict["sat_lat"]) == len(val)
        for val in shared_dict.items()
        if isinstance(val, ndarray)
    ), "SAR - Not all fields the same length as 'sat_lat' or 'sat_lon'"

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
    success, err_str = thisalg.process(l1b, shared_dict)  # filter area

    assert success, f"SIN - Algorithm failed due to: {err_str}"

    # check if all the lats and lons are within the target area
    assert all(
        (
            config["shared"]["min_longitude"] <= lon <= config["shared"]["max_longitude"]
            and config["shared"]["min_latitude"] <= lat <= config["shared"]["max_latitude"]
        )
        for lon, lat in zip(shared_dict["sat_lon"], shared_dict["sat_lat"])
    ), "SIN - Contains lat-lon values outside of area filter range"

    # check if all fields from the file have been filtered to the same size as sat_lat
    assert all(
        len(shared_dict["sat_lat"]) == len(val)
        for val in shared_dict.items()
        if isinstance(val, ndarray)
    ), "SIN - Not all fields the same length as 'sat_lat' or 'sat_lon'"
