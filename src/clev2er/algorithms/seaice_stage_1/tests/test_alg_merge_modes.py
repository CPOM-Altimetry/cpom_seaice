"""pytest for algorithm
clev2er.algorithms.seaice_stage_1.alg_merge_modes
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.algorithms.seaice_stage_1.alg_add_si_conc import Algorithm as AddSIConc
from clev2er.algorithms.seaice_stage_1.alg_area_filter import Algorithm as AreaFilter
from clev2er.algorithms.seaice_stage_1.alg_crop_waveform import (
    Algorithm as CropWaveform,
)
from clev2er.algorithms.seaice_stage_1.alg_cs2_wave_discrimination import (
    Algorithm as WaveDiscrimination,
)
from clev2er.algorithms.seaice_stage_1.alg_elev_calculations import (
    Algorithm as ElevCalc,
)
from clev2er.algorithms.seaice_stage_1.alg_flag_filters import Algorithm as FlagFilter
from clev2er.algorithms.seaice_stage_1.alg_giles_retrack import (
    Algorithm as GilesRetrack,
)
from clev2er.algorithms.seaice_stage_1.alg_ingest_cs2 import Algorithm as IngestCS2
from clev2er.algorithms.seaice_stage_1.alg_merge_modes import Algorithm
from clev2er.algorithms.seaice_stage_1.alg_pulse_peakiness import (
    Algorithm as PulsePeakiness,
)
from clev2er.algorithms.seaice_stage_1.alg_smooth_waveform import (
    Algorithm as SmoothWaveform,
)
from clev2er.algorithms.seaice_stage_1.alg_threshold_retrack import (
    Algorithm as ThresholdRetrack,
)
from clev2er.utils.config.load_config_settings import load_config_files

logger = logging.getLogger(__name__)

# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals


@pytest.fixture
def config() -> dict:
    """Pytest fixture for the config dictionary

    Returns:
        dict: config dictionary
    """
    # load config
    chain_config, _, _, _, _ = load_config_files("seaice_stage_1")

    # Set to Sequential Processing
    chain_config["chain"]["use_multi_processing"] = False
    chain_config["alg_merge_modes"][
        "merge_file_dir"
    ] = "/home/jgnq4/Documents/sea_ice_processor/test_files/feb_2024/elev_merge_test"
    return chain_config


@pytest.fixture
def previous_steps(
    config: Dict,
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
            "smooth_waveform": SmoothWaveform(config, logger),
            "threshold_retack": ThresholdRetrack(config, logger),
            "giles_retrack": GilesRetrack(config, logger),
            "elev_calculations": ElevCalc(config, logger),
        }
    except KeyError as exc:
        raise RuntimeError(f"Could not initialize previous steps in chain {exc}") from exc

    return chain_previous_steps


@pytest.fixture
def thisalg(config: Dict) -> Algorithm:
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


def test_merge_modes(
    previous_steps: Dict,
    thisalg: Algorithm,
) -> None:
    """test alg_sla_calculations.py for SAR waves

    Test plan:
    Load an SAR file
    run Algorithm.process() on each
    test that the files return (True, "")
    """

    base_dir = Path(os.environ["CLEV2ER_BASE_DIR"])
    assert base_dir is not None

    assert (
        len(os.listdir(thisalg.merge_file_dir)) == 0
    ), """NOT AN ALGORITHM FAIL - Test directory not empty! 
    Please empty this as files will be appended :)"""

    # ================================== SAR FILE TESTING ==========================================

    # load SAR file
    l1b_sar_files = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sar").glob("*.nc")
    )
    l1b_sin_fles = list(
        (base_dir / "testdata" / "cs2" / "l1bfiles" / "arctic" / "sin").glob("*.nc")
    )

    all_files = l1b_sar_files + l1b_sin_fles

    orbit_numbers: Dict[
        int, int
    ] = {}  # Dict for orbit numbers as keys and number of samples as values

    for fi, l1b_file in enumerate(all_files):
        logger.info(" ------ %d -------", fi + 1)
        try:
            l1b = Dataset(l1b_file)
            logger.info("Loaded %s", l1b_file)
        except IOError:
            assert False, f"{l1b_file} could not be read"

        shared_dict: Dict[str, Any] = {}

        for title, step in previous_steps.items():
            success, err_str = step.process(l1b, shared_dict)  # type: ignore[attr-defined]
            if not success:
                logger.error("SAR - Error with previous step: %s\n%s", title, err_str)

        success, err_str = thisalg.process(l1b, shared_dict)

        n_records = orbit_numbers.get(l1b.rel_orbit_number, 0)
        orbit_numbers.update(
            {l1b.rel_orbit_number: n_records + shared_dict["measurement_time"].size}
        )

        assert success, f"Algorithm failed when processing {l1b_file} due to: {err_str}"

    # Algorithm tests
    for o_n, n_s in orbit_numbers.items():
        # check that a file exists for each orbit number in the l1b files
        assert os.path.exists(
            os.path.join(thisalg.merge_file_dir, f"merge_{o_n:04d}.nc")
        ), "Merge file does not exist"

        merge_nc = Dataset(os.path.join(thisalg.merge_file_dir, f"merge_{o_n:04d}.nc"), mode="r")
        actual_n_samples = merge_nc["measurement_time"][:].size

        assert (
            actual_n_samples == n_s
        ), "Number of samples in merge file doesn't match estimated number"
