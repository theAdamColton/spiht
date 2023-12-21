import math
from typing import List
import numpy as np
from pywt import wavedec2, waverec2
import pywt
from collections import namedtuple

from .utils import imshow, is_power_of_two

def is_bit_set(x, n):
    return np.bitwise_and(x, 2**n)

def is_element_significant(x, n):
    return np.any(np.abs(x) >= 2**n)

def get_offspring(i,j,h,w):
    if 2*i >= h or 2*j >= w:
        return []
    return [
            (2*i,2*j), 
            (2*i, 2*j+1),
            (2*i+1, 2*j),
            (2*i+1, 2*j+1),
        ]

def is_set_significant(arr,i,j,n):
    _, h, w =arr.shape

    if is_element_significant(arr[:,i,j], n):
        return True

    for i,j in get_offspring(i,j,h,w):
        if is_set_significant(arr,i,j,n):
            return True

    return False

LspElement = namedtuple("LspElement", ['i','j','n'])
LisElement = namedtuple("LisElement", ['i', 'j', 'type'])

def spiht_encode(image, wavelet='bior4.4', level=6, max_bits=100000):
    coeffs = wavedec2(image, mode='periodization', wavelet=wavelet, level=level)
    arr,slices = pywt.coeffs_to_array(coeffs, padding=None, axes=(-2,-1))
    arr = arr.astype(np.int16)

    c, h, w = arr.shape

    lis = [LisElement(i,j,'A') for i in range(h) for j in range(w) if i%2 != 0 and j%2!=0 ]
    lip = [(i,j) for i in range(h) for j in range(w)]

    lsp:List[LspElement] = []

    out = []

    def append_to_out(*x):
        for y in x:
            if len(out) >= max_bits:
                return out
            out.append(y)

    n = math.floor(math.log2(np.abs(arr).max()))

    while n >=0:
        # sorting pass

        lsp_len = len(lsp)

        for i,j in lip:
            is_element_sig = is_element_significant(arr[:, i, j], n)

            append_to_out(is_element_sig)

            if is_element_sig:
                # outputs sign
                append_to_out( x for x in arr[:,i,j]>0)
                lsp.append(LspElement(i,j,n))

        while len(lis) > 0:
            lis_element = lis.pop(0)

            i,j = lis_element.i, lis_element.j

            if lis_element.type == "A":
                is_set_sig = is_set_significant(arr,lis_element.i, lis_element.j,n)

                append_to_out(is_set_sig)

                if is_set_sig:
                    for k,l in get_offspring(lis_element.i, lis_element.j, h, w):
                        is_element_sig = is_element_significant(arr[:, k, l], n)
                        append_to_out(is_element_sig)

                        if is_element_sig:
                            lsp.append(LspElement(k,l,n))
                            append_to_out(x for x in arr[:, k,l] > 0)
                        else:
                            lip.append((k,l))

                    has_descendents_past_offspring = i * 4 < h and j * 4 < w

                    if has_descendents_past_offspring:
                        lis.append(LisElement(i,j,"B"))
                            
            else:
                # type B

                descendents_past_offspring = sum([get_offspring(k,l,h,w) for (k,l) in get_offspring(i,j,h,w)], [])
                is_l_significant = any(is_set_significant(arr,k,l,n) for (k,l) in descendents_past_offspring)
                append_to_out(is_l_significant)
                if is_l_significant:
                    for k,l in descendents_past_offspring:
                        lis.append(LisElement(k,l,"A"))


        # refinement pass
        print('refinement')
        for lsp_i in range(lsp_len):
            i,j,lsp_n = lsp[lsp_i]
            if lsp_n < n:
                bits = is_bit_set(arr[:,i,j], n)
                append_to_out(x for x in bits)

        # quantization pass 
        print(f'quant pass n {n} kb:{len(out)/1000:.2f}')
        n -= 1

    return out, n, h, w

def spiht_decode(d, n, h, w, wavelet='bior4.4', level=6):
    c = 3
    arr = np.zeros(c,h,w)

    lis = [LisElement(i,j,'A') for i in range(h) for j in range(w) if i%2 != 0 and j%2!=0 ]
    lip = [(i,j) for i in range(h) for j in range(w)]

    lsp:List[LspElement] = []

    def ret():
        # does this just to get the coeff slices
        dummy_image = arr
        dummy_coeffs = wavedec2(dummy_image, mode='periodization', wavelet=wavelet, level=level)
        _, slices = pywt.coeffs_to_array(dummy_coeffs, padding=None, axes=(-2,-1))
        coeffs = pywt.array_to_coeffs(arr, slices)
        rec_image = pywt.waverec2(coeffs, mode='periodization', wavelet=wavelet)
        return rec_image

    def pop():
        if len(d) > 0:
            return d.pop(0)
        raise StopIteration()

    try:
        while n >=0:
            # sorting pass

            lsp_len = len(lsp)

            for i,j in lip:
                is_element_sig = pop()

                if is_element_sig:
                    signs = np.ndarray([pop() for _ in range(c)]) - 1
                    arr[:, i,j] = arr[:, i, j] * signs[:, None, None]


            while len(lis) > 0:
                lis_element = lis.pop(0)

                i,j = lis_element.i, lis_element.j

                if lis_element.type == "A":
                    is_set_sig = pop()

                    if is_set_sig:
                        for k,l in get_offspring(lis_element.i, lis_element.j, h, w):
                            is_element_sig = pop()

                            if is_element_sig:
                                lsp.append(LspElement(k,l,n))
                                # 2^n <= abs(arr[k,l]) <= 2^(n+1)
                                values = np.zeros(c) + 1.5
                                values = values * 2**n

                                # either 1 or -1
                                signs = np.ndarray([pop() for _ in range(c)]) * 2 - 1

                                arr[:, k,l] = arr[:, k, l] + values * signs[:, None, None]
                            else:
                                lip.append((k,l))

                        has_descendents_past_offspring = i * 4 < h and j * 4 < w

                        if has_descendents_past_offspring:
                            lis.append(LisElement(i,j,"B"))
                                
                else:
                    # type B

                    descendents_past_offspring = sum([get_offspring(k,l,h,w) for (k,l) in get_offspring(i,j,h,w)], [])
                    is_l_significant = any(is_set_significant(arr,k,l,n) for (k,l) in descendents_past_offspring)
                    is_l_significant = pop()
                    if is_l_significant:
                        for k,l in descendents_past_offspring:
                            lis.append(LisElement(k,l,"A"))


            # refinement pass
            print('refinement')
            for lsp_i in range(lsp_len):
                i,j,lsp_n = lsp[lsp_i]
                if lsp_n < n:
                    bits = np.ndarray([pop() for _ in range(c)])

                    # either 1 or -1
                    signs = (arr[:,i,j] > 0) * 2 - 1

                    values = bits * 1.5
                    # either 0 or 1.5 * 2**n
                    values = values * 2**n

                    arr[:,i,j] = arr[:,i,j] + values * signs

            # quantization pass 
            n -= 1

    except StopIteration:
        return ret()

    return ret()

