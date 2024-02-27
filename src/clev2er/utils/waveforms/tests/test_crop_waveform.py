"""pytest for utils function
    clev2er.utils.waveforms.crop_waveform
"""

import logging

import numpy as np

from clev2er.utils.waveforms.crop_waveform import crop_waveform

logger = logging.getLogger(__name__)


def test_crop_waveform() -> None:
    """test crop_waveform.py

    Test plan:
    Make an array of zeroes and set one value to 1
    Use crop_wavform on array of zeroes
    Check that waveform is the right length
    Check that 1 value is at position n
    """

    logger.info("Testing crop_wavform function")

    test_arr = np.zeros(100, dtype=int)
    test_arr[36] = 1

    cropped_arr = crop_waveform(test_arr, 5, 15)

    assert cropped_arr.size == 15, f"Length of cropped array is {cropped_arr.size}, not 15"
    logger.info("Cropped array is the right size")

    assert np.argmax(cropped_arr) == 5, f"Max value is at {np.argmax(cropped_arr)}, not at pos 5"
    logger.info("Cropped array contains max value at the correct index")
