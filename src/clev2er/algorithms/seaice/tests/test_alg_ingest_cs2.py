"""pytest for algorithm
   clev2er.algorithms.seaice.alg_ingest_cs2.py
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice.alg_ingest_cs2 import Algorithm
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)


def test_alg_ingest_cs2() -> None:
    """test alg_ingest_cs2.py
    Load an SAR and an SARIn files
    run Algorithm.process() on each
    test that the files return (True, "")
    check if arrays were loaded into shared_dict
    check if arrays are all the same length
    """

    ingested_fields = [
        "sat_lat",
        "sat_lon",
        "measurement_time",
        "sat_altitude",
        "window_delay",
        "waveform",
        "waveform_ssd",
        "dry_trop_correction",
        "wet_trop_correction",
        "inv_baro_correction",
        "iono_correction",
        "ocean_tide",
        "long_period_tide",
        "loading_tide",
        "earth_tide",
        "pole_tide",
        "surface_type",
    ]

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    # load config
    config, _, _, _, _ = load_config_files("seaice")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    # Initialise the Algorithm
    try:
        thisalg = Algorithm(config, logger)  # no config used for this alg
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    # TESTING WITH SAR FILE

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
    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SAR - Algorithm failed due to: {err_str}"
    assert all(
        field in shared_dict for field in ingested_fields
    ), "SAR - Not all fields in shared_dict"
    assert all(
        len(shared_dict[ingested_fields[0]]) == len(shared_dict[key]) for key in ingested_fields
    ), "SAR - Not all fields the same length"
    assert shared_dict["instr_mode"] == "SAR", "SAR - Did not correctly identify instrument mode"

    # TESTING WITH SIN FILE

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

    shared_dict.clear()
    success, err_str = thisalg.process(l1b, shared_dict)

    assert success, f"SIN - Algorithm failed due to: {err_str}"
    assert all(
        field in shared_dict for field in ingested_fields
    ), "SIN - Not all fields in shared_dict"
    assert all(
        len(shared_dict[ingested_fields[0]]) == len(shared_dict[key]) and len(shared_dict[key]) != 0
        for key in ingested_fields
    ), "SIN - Not all fields the same length"
    assert shared_dict["instr_mode"] == "SIN", "SIN - Did not correctly identify instrument mode"
