"""pytest for utils functions
clev2er.utis.io.ingest_l1b
"""

import logging
import os

from netCDF4 import Dataset  # pylint:disable=no-name-in-module

from clev2er.utils.io.ingest_l1b import unpack

logger = logging.getLevelName(__name__)


def test_unpack() -> None:
    """Tests the unpack() function

    Test plan:
    Open testdata l1b file
    unpack a 20Hz variable
    unpack a 1Hz variable
    test that they are the same length
    """

    test_file = os.path.join(
        os.environ["CLEV2ER_BASE_DIR"],
        "testdata/cs2/l1bfiles/arctic/sar",
        "CS_OFFL_SIR_SAR_1B_20221214T001528_20221214T002325_E001.nc",
    )

    nc_file = Dataset(test_file)

    var_1 = unpack("alt_20_ku", nc_file, "time_20_ku", "time_cor_01")
    var_2 = unpack("load_tide_01", nc_file, "time_20_ku", "time_cor_01")

    assert var_1.size == var_2.size, "Unpack did not extrapolate 1Hz variable to 20Hz"
