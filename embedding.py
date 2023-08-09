import numpy as np

from stego import str_to_bits, bits_to_str
import math
from stego.floatbits import boolbin_to_float, float_to_binary
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def binzoo_to_floatzoo(zoo, n_jobs=-1):
    def binmat_to_floatmat(mat):
        return [boolbin_to_float(bin[0], tuple(bin[1:9]), tuple(bin[9:])) for bin in mat]
    return np.array(Parallel(n_jobs=n_jobs)(delayed(binmat_to_floatmat)(mat) for mat in zoo))

def modify(weights, malware_name = 'malware_2448b', fill=True, lsb=23, msb=False):
    with open(os.path.join('data', 'malware', malware_name), 'r') as malware_file:
        malware_str = malware_file.read()
    
    weights_embedded_bin = embed_s(weights, malware_str, lsb=lsb, fill=fill, inplace=False, msb=msb)
    return binzoo_to_floatzoo(weights_embedded_bin)

def embed_s(host, s: str, lsb=23, fill=True, inplace=True, msb=False):
    def return_reshaped():
        if not fill:
            n = len(s_bits)//lsb
            complete_n = n*lsb
        else:
            slice_len = (host_bits_capacity//8)*8
            n = slice_len//lsb
            complete_n = n*lsb
        reshaped = s_bits[:complete_n].reshape(n,lsb)
        if fill:
            remainder=np.array([])
        else:
            remainder = s_bits[complete_n:]
        
        return reshaped, remainder
    if len(host.shape)==2:
        host = np.array([[float_to_binary(v) for v in arr] for arr in host])
        inplace=True

    assert 1 <= lsb <= 23, f"lsb must be between 1 and 23, lsb: {lsb}"
    
    host_bits_capacity = host.shape[1]*lsb
    s_bits = np.array(str_to_bits(s))
    num_bits_s = len(s_bits)
    if fill:
        dupe_amount = math.ceil(host_bits_capacity/num_bits_s)
        s_bits = np.tile(s_bits, dupe_amount)
    
    reshaped, remainder = return_reshaped()

    if not fill and len(s_bits)>host_bits_capacity:
        fill=True
        reshaped, remainder = return_reshaped()
            
    a, b, c = host.shape
    x, y = reshaped.shape
    if fill:
        assert x <= b and y <= c, f"reshaped array doesn't fit in the host array, lsb: {lsb}, reshaped.shape: {reshaped.shape:}"
    else:
        assert x < b and y < c, f"reshaped array doesn't fit in the host array, lsb: {lsb}, reshaped.shape: {reshaped.shape:}"
    if inplace:
        host_embedded=host
    else:
        host_embedded = np.copy(host)
    if msb:
        host_embedded[:, 0:x, 9:9+lsb] = reshaped
        if not fill:
            host_embedded[:, x:x+1, 9:9+lsb]=remainder
    else:
        host_embedded[:, 0:x, c-y:] = reshaped
        if not fill:
            host_embedded[:, x:x+1, c-len(remainder):]=remainder
    
    return host_embedded

def retrieve_s(host, s_len, binary=False, lsb=23, fill=True):
    s_bits_amount = s_len*8 if not binary else s_len
    
    if len(host.shape)==2:
        sample=host
    else:
        sample=host[0]
        
    if fill:
        block = sample[:,32-lsb:].ravel()
        return bits_to_str(block[:s_bits_amount])
    else:
        n = s_bits_amount//lsb
        complete_n = n*lsb
        remainder_len = s_bits_amount-complete_n
        
        reshaped_embedded_s = sample[:n,32-lsb:]
        remainder_embdded_s = sample[n:n+1,32-remainder_len:]

        embedded_s = list(reshaped_embedded_s.ravel())+list(remainder_embdded_s.ravel())
        return bits_to_str(embedded_s)

if __name__ == '__main__':
    fill=True
    msb = False
    malware_name = 'malware_12584bytes'
    
    mnist_zoo_bin_array = np.load(f'{NNSTEGO_DATA_ZOO}/mnist/zoo_weights_bin.npz')['arr_0']
    with open(f'./malware/{malware_name}', 'rb') as malware_file:
        malware_str = malware_file.read()

    exports = {}
    for lsb in tqdm(range(23,0,-1)):
        host_embedded_bin = embed_s(mnist_zoo_bin_array, malware_str, lsb=lsb, fill=fill, inplace=False, msb=msb)

        host_embedded = binzoo_to_floatzoo(host_embedded_bin)
        exports[str(lsb)] = host_embedded
    np.savez_compressed(f'{NNSTEGO_DATA}/embedded/embedded_{malware_name}{"_filled" if fill else ""}{"_msb" if msb else ""}', **exports)