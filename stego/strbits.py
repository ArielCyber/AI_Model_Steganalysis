from typing import List, Union

float_map = {'1': True, '0': False}
def str_to_bits(s: str, as_list: bool = False, boolbits=True) -> Union[List[int], List[List[int]]]:
    byte_str = bytes(s, "ascii") if type(s) != bytes else s
    tmp = [bin(b)[2:].rjust(8, "0") for b in byte_str]
    # for b in bytes(s, "ascii"):
    #     s_bin = bin(b)[2:].rjust(8, "0")
    #     tmp.append(s_bin)
    if as_list:
        return [list(map(int, list(x))) for x in tmp]
    if boolbits:
        return [float_map[x] for x in "".join(tmp)]
    else:
        return [int(x) for x in "".join(tmp)]


def bits_to_str(b: List[int]) -> str:
    tmp = []
    for i in range(0, len(b), 8):
        c = chr(int("".join(map(str, b[i:i + 8])), 2))
        tmp.append(c)
    return "".join(tmp)

def str_to_bitstring(s: str) :
    return ''.join([bin(ord(char))[2:].zfill(8) for char in s])