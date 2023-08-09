import ctypes
from typing import Tuple
import numpy as np
def float_to_binary(num: float, bool=True):
    """
    Convert float value to a binary representation - 1s and 0s
    (True is 1 and False is 0)

    Args:
        num: float value

    Returns:
        Tuple[bool]: tuple of bools
    """
    if bool:
        float_map = {'1': True, '0': False}
    else:
        float_map = {'1': 1, '0': 0}
    bin_num = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
    bin_num = bin_num.rjust(32, "0")
    return np.array([float_map[x] for x in bin_num])

def binstr_to_float(binstr: str) -> float:
    assert len(binstr) == 32

    sign = binstr[0]
    exponent = tuple(binstr[1:9])
    fraction = tuple(binstr[9:32])

    return charbin_to_float(sign, exponent, fraction)

def charbin_to_float(sign: str, exponent: Tuple[str], fraction: Tuple[str]) -> float:
        """
        From the binary float representation "reconstruct" (calculate) the float value

        Args:
            sign: sign value as a bool
            exponent: exponent as a tuple of bools
            fraction: fraction as a tuple of bools

        Returns:
            float: calculated float value
        """
        # print(sign)
        # print(exponent)
        # print(fraction)
        if len(fraction) != 23:
            raise ValueError("Fraction should be 23 values bits")
        if len(exponent) != 8:
            raise ValueError("Exponent should be 8 bits")
        # if not isinstance(sign, bool):
        #     raise ValueError("Sign is a single bit")

        fraction = ('1',) + fraction
        fraction_as_int = int("".join(fraction), 2) / 2**23
        exponent_as_int = int("".join(exponent), 2) - 127
        sign_as_int: int = -1 if sign=='1' else 1
        return sign_as_int * fraction_as_int * (2**exponent_as_int)

def boolbin_to_float(sign: bool, exponent: Tuple[bool], fraction: Tuple[bool]) -> float:
        """
        From the binary float representation "reconstruct" (calculate) the float value

        Args:
            sign: sign value as a bool
            exponent: exponent as a tuple of bools
            fraction: fraction as a tuple of bools

        Returns:
            float: calculated float value
        """

        if len(fraction) != 23:
            raise ValueError("Fraction should be 23 values bits")
        if len(exponent) != 8:
            raise ValueError("Exponent should be 8 bits")
        if not isinstance(sign, np.bool_):
            #print(type(sign))
            raise ValueError("Sign is a single bit")

        fraction = (True,) + fraction
        fraction_as_int = int(FloatBinary.bool_to_str_representation(fraction), 2) / 2**23
        exponent_as_int = int(FloatBinary.bool_to_str_representation(exponent), 2) - 127
        sign_as_int: int = -1 if sign else 1
        return sign_as_int * fraction_as_int * (2**exponent_as_int)

class FloatBinary:

    def __init__(self, v: float) -> None:
        self.v: float = float(v)

        # Binary representation of the float value LSB
        self._bin: Tuple[bool] = self.float_to_binary(self.v)

        # Binary representation splitted to sign, exponent, fraction/mantissa
        self._splitted_bin: dict = self.split_binary(self._bin)

        # 1 bit for sign
        self.sign: bool = self._splitted_bin["sign"]

        # 8 bits for exponent
        self.exponent: Tuple[bool] = self._splitted_bin["exponent"]

        # Full 23 bits for fraction/mantissa - 3 parts: first 7 bits, middle 8 bits, last 8 bits
        self.fraction: Tuple[bool] = self._splitted_bin["fraction"]

    def float_to_binary(self, num: float):
        """
        Convert float value to a binary representation - 1s and 0s
        (True is 1 and False is 0)

        Args:
            num: float value

        Returns:
            Tuple[bool]: tuple of bools
        """

        bin_num = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
        bin_num = bin_num.rjust(32, "0")
        return tuple([False if x == "0" else True for x in list(bin_num)])

    @staticmethod
    def split_binary(binary_float: Tuple[bool]) -> dict:
        """
        Structures (splits) the binary representation to sign, exponent, fraction

        Args:
            binary_float: binary representation of the float value

        Returns:
            dict: part name and binary representation
        """

        if len(binary_float) != 32:
            raise ValueError("You need 32 bits")

        sign = binary_float[0]
        exponent = binary_float[1:9]
        fraction = binary_float[9:32]
        d = {"sign": sign, "exponent": exponent, "fraction": fraction}
        return d

    @staticmethod
    def bin_to_float(sign: bool, exponent: Tuple[bool], fraction: Tuple[bool]) -> float:
        """
        From the binary float representation "reconstruct" (calculate) the float value

        Args:
            sign: sign value as a bool
            exponent: exponent as a tuple of bools
            fraction: fraction as a tuple of bools

        Returns:
            float: calculated float value
        """

        if len(fraction) != 23:
            raise ValueError("Fraction should be 23 values bits")
        if len(exponent) != 8:
            raise ValueError("Exponent should be 8 bits")
        if not isinstance(sign, bool):
            raise ValueError("Sign is a single bit")

        fraction = (True,) + fraction
        fraction_as_int = int(FloatBinary.bool_to_str_representation(fraction), 2) / 2**23
        exponent_as_int = int(FloatBinary.bool_to_str_representation(exponent), 2) - 127
        sign_as_int: int = -1 if sign else 1
        return sign_as_int * fraction_as_int * (2**exponent_as_int)

    def reconstruct_float_value(self) -> float:
        """
        From the binary float representation "reconstruct" (calculate) the float value

        Returns:
            float: calculated float value
        """

        return self.bin_to_float(self.sign, self.exponent, self.fraction)

    @staticmethod
    def bool_to_str_representation(vals) -> str:
        # 0 = False; 1 = True
        return "".join(["1" if x else "0" for x in vals])

    def modify_clone(self,
                     sign: bool = None,
                     exponent: Tuple[bool] = None,
                     fraction: Tuple[bool] = None) -> "FloatBinary":
        sign = sign if sign is not None else self.sign
        exponent = exponent if exponent is not None else self.exponent
        fraction = fraction if fraction is not None else self.fraction
        
        new_float_value = FloatBinary.bin_to_float(sign=sign, exponent=exponent, fraction=fraction)
        return FloatBinary(new_float_value)

    def as_bit_str(self) -> str:
        return self.bool_to_str_representation(self._bin)
