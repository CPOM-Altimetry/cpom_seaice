"""pytest for algorithm
    clev2er.algorithms.seaice.alg_crop_waveform
"""

import logging

import numpy as np

from clev2er.utils.waveforms.peakiness import peakiness

logger = logging.getLogger(__name__)


def test_peakiness() -> None:
    """test peakiness.py

    Test plan:
    Run peakiness on sample waveform that has a known peakiness value
    Check if result is a float
    Check if result is close to the known value
    """

    logger.info("Testing peakiness function")

    # sample wave taken from SAR file for 01/01/2018 at 03:22:21 (index 0 after filtering)
    test_wave = np.array(
        [
            8.0,
            7.0,
            8.0,
            8.0,
            8.0,
            10.0,
            10.0,
            12.0,
            8.0,
            12.0,
            8.0,
            14.0,
            8.0,
            16.0,
            9.0,
            19.0,
            9.0,
            22.0,
            8.0,
            25.0,
            9.0,
            29.0,
            9.0,
            34.0,
            9.0,
            39.0,
            10.0,
            46.0,
            10.0,
            54.0,
            11.0,
            68.0,
            11.0,
            85.0,
            15.0,
            109.0,
            17.0,
            144.0,
            25.0,
            203.0,
            34.0,
            303.0,
            60.0,
            503.0,
            111.0,
            928.0,
            260.0,
            3025.0,
            1299.0,
            22106.0,
            65535.0,
            39020.0,
            5172.0,
            5757.0,
            1510.0,
            1482.0,
            798.0,
            870.0,
            476.0,
            619.0,
            332.0,
            333.0,
            232.0,
            332.0,
            187.0,
            224.0,
            198.0,
            221.0,
            168.0,
            164.0,
            137.0,
            128.0,
            106.0,
            105.0,
            90.0,
            83.0,
            71.0,
            60.0,
            63.0,
            64.0,
            77.0,
            71.0,
            63.0,
            56.0,
            67.0,
            51.0,
            58.0,
            49.0,
            61.0,
            61.0,
            71.0,
            60.0,
            57.0,
            42.0,
            47.0,
            40.0,
            56.0,
            51.0,
            54.0,
            45.0,
            54.0,
            47.0,
            48.0,
            34.0,
            39.0,
            31.0,
            39.0,
            35.0,
            45.0,
            24.0,
            32.0,
            27.0,
            27.0,
            22.0,
            26.0,
            19.0,
            24.0,
            18.0,
            20.0,
            18.0,
            22.0,
            15.0,
            23.0,
            23.0,
            24.0,
            23.0,
            20.0,
            17.0,
        ]
    )

    pp = peakiness(test_wave, 10, 20)

    logger.info("Returned value - %f (Expected for this sample = 44.512216)", pp)

    # check that the returned value is a float
    assert isinstance(pp, float), f"Peakiness returned a {type(pp)}, should be a float"

    # check that the function returns a value close to what is expected from the sample
    assert np.isclose(
        pp, 44.5, atol=0.2
    ), f"Value {pp} is not close to 44.5 within tolerance of 0.2"
