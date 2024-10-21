"""Module for utilities involving binary flags

check_bits(): Checks the byte at a given index is set in an array of binary representations
to_binary(): Converts an array of integers to binary representations

Author: Ben Palmer
Date: 21 October 2024
"""

import numpy as np
import numpy.typing as npt


# Numpy didn't like it when I used apply_along_axis
def check_bit(array: npt.NDArray[np.str_], bit_index: int) -> npt.NDArray[np.bool_]:
    """Checks wether the bit at bit_index is set

    Args:
        array (np.ndarray): Array of binary representations as strings
        bit_index (int): Index of the bit to check (NOT ARRAY INDEX!)
    """

    if 0 >= bit_index <= len(array[0]):
        raise ValueError("Index needs to be between 0 and width of binary.")

    if len(array.shape) > 1:
        raise ValueError(
            f"Input array has too many dimensions! Expected 1, got {len(array.shape)}."
        )
    if array.dtype.type != np.str_:
        raise ValueError(
            f"Input array is not of the correct data type. Expected string, got {array.dtype}"
        )

    out_arr = np.zeros(array.shape, dtype=bool)

    for i in range(array.shape[0]):
        out_arr[i] = int(array[i][len(array[i]) - bit_index])

    return out_arr


# Numpy didn't like it when I used apply_along_axis
def to_binary(arr: npt.NDArray[np.int_], width: int = 32) -> npt.NDArray[np.str_]:
    """Converts a 1D numpy.ndarray of ints to an array of binary representations as strings

    Args:
        arr (np.ndarray): Input array of integers
        width (int, optional): Width of the binary representation. Defaults to 32.

    Returns:
        np.ndarray[str]: binary representations
    """
    if len(arr.shape) > 1:
        raise ValueError(f"Input array has too many dimensions! Expected 1, got {len(arr.shape)}.")
    if arr.dtype.kind != "i":
        raise ValueError(
            f"Input array is not of the correct data type. Expected int, got {arr.dtype}"
        )

    out_arr = np.zeros(arr.shape, dtype="<U32")

    for i in range(arr.shape[0]):
        out_arr[i] = np.binary_repr(arr[i], width=width)

    return out_arr
