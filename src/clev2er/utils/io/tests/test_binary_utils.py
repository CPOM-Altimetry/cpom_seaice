"""pytest for utils functions
clev2er.utils.io.binary_utils
"""

import numpy as np

from clev2er.utils.io.binary_utils import check_bit, to_binary


def test_to_binary() -> None:
    """pytest for to_binary

    Plan:
        Make array of ints
        Convert to binary
        Check if binaries match expected values
    """
    in_int = np.array([1, 2, 4, 8])
    binaries = to_binary(in_int, 4)

    bin_2 = np.array(["0001", "0010", "0100", "1000"])
    assert (binaries == bin_2).all(), "to_binary did not properly convert all values to binary"


def test_check_bit() -> None:
    """pytest for check_bit

    Plan:
        Make array of binaries
        Check a bit
        Check if output is as expected for all values
    """
    in_bins = np.array(["0001", "0010", "0100", "1000"])

    bit_2 = check_bit(in_bins, 2)
    expected_res = np.array([False, True, False, False])

    assert (bit_2 == expected_res).all(), "check_bit did not properly check the given binaries"
