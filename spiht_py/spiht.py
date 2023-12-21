import math
from typing import List
import numpy as np
from pywt import wavedec2, waverec2
import pywt
from collections import namedtuple

from .utils import imshow, is_power_of_two

def is_bit_set(x, n):
    return np.bitwise_and(np.abs(x), 2**n) > 0

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

def are_descendants_significant(arr,i,j,n):
    _, h, w =arr.shape

    for i,j in get_offspring(i,j,h,w):
        if is_set_significant(arr,i,j,n):
            return True

    return False

def is_set_significant(arr,i,j,n):
    _, h, w =arr.shape

    if is_element_significant(arr[:,i,j], n):
        return True

    for i,j in get_offspring(i,j,h,w):
        if is_set_significant(arr,i,j,n):
            return True

    return False

LisElement = namedtuple("LisElement", ['i', 'j', 'type'])

Q_SCALE = 100
def quantize(arr):
    return (arr*Q_SCALE).astype(np.int16)

def dequantize(arr):
    return arr / Q_SCALE

def spiht_encode(image, wavelet='bior4.4', level=6, max_bits=200000):
    coeffs = wavedec2(image, mode='periodization', wavelet=wavelet, level=level)
    arr,slices = pywt.coeffs_to_array(coeffs, padding=None, axes=(-2,-1))

    c, h, w = arr.shape
    arr = quantize(arr)

    n = math.floor(math.log2(np.abs(arr).max()))
    max_n = n


    class __EndEncoding(Exception):
        pass

    curr_i = dict(i=0)
    out = [0] * max_bits
    def append_to_out(*x):
        i = curr_i['i']
        for y in x:
            if i >= max_bits:
                raise __EndEncoding()
            out[i] = y
            i += 1
        curr_i['i'] = i

    ll_h = slices[0][1].stop
    ll_w = slices[0][2].stop
    lis = []
    for i in range(ll_h):
        for j in range(ll_w):
            if i % 2 == 0 and j% 2 == 0:
                continue
            lis.append(LisElement(i,j,'A'))
    lip = [(i,j) for i in range(ll_h) for j in range(ll_w)]
    lsp= []


    try:
        while n >=0:
            # sorting pass

            lsp_len = len(lsp)

            for i,j in lip:
                is_element_sig = is_element_significant(arr[:, i, j], n)

                append_to_out(is_element_sig)

                if is_element_sig:
                    # outputs all signs
                    append_to_out(*tuple(x for x in arr[:,i,j]>0))

                    # outputs all significances
                    append_to_out(*tuple(is_element_significant(x,n) for x in arr[:,i,j]))

                    lsp.append((i,j))

            while len(lis) > 0:
                lis_element = lis.pop(0)

                i,j = lis_element.i, lis_element.j

                if lis_element.type == "A":
                    is_set_sig = are_descendants_significant(arr,lis_element.i, lis_element.j,n)

                    append_to_out(is_set_sig)

                    if is_set_sig:
                        for k,l in get_offspring(lis_element.i, lis_element.j, h, w):
                            is_element_sig = is_element_significant(arr[:, k, l], n)
                            append_to_out(is_element_sig)

                            if is_element_sig:
                                lsp.append((k,l))
                                # outputs all signs
                                append_to_out(*tuple(x for x in arr[:, k,l] > 0))

                                # outputs all significances
                                append_to_out(*tuple(is_element_significant(x,n) for x in arr[:, k, l]))
                            else:
                                lip.append((k,l))

                        has_descendents_past_offspring = i * 4 < h and j * 4 < w

                        if has_descendents_past_offspring:
                            lis.append(LisElement(i,j,"B"))
                                
                else:
                    # type B
                    descendents_past_offspring = sum([get_offspring(k,l,h,w) for (k,l) in get_offspring(i,j,h,w)], [])
                    is_l_significant = any(are_descendants_significant(arr,k,l,n) for (k,l) in descendents_past_offspring)
                    append_to_out(is_l_significant)
                    if is_l_significant:
                        for k,l in descendents_past_offspring:
                            lis.append(LisElement(k,l,"A"))


            # refinement pass
            print('refinement')

            for lsp_i in range(lsp_len):
                i,j = lsp[lsp_i]
                bits = is_bit_set(arr[:,i,j], n)
                append_to_out(*tuple(x for x in bits))

            # quantization pass 
            print(f'quant pass n {n} kb:{curr_i["i"]/1000:.2f}')
            n -= 1
    except __EndEncoding:
        return out, max_n, h, w

    return out, max_n, h, w

def spiht_decode(d, n, h, w, wavelet='bior4.4', level=6):
    c = 3
    arr = np.zeros((c,h,w))


    # does this just to get the coeff slices
    dummy_image = arr
    dummy_coeffs = wavedec2(dummy_image, mode='periodization', wavelet=wavelet, level=level)
    _, slices = pywt.coeffs_to_array(dummy_coeffs, padding=None, axes=(-2,-1))

    def ret():
        coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
        rec_image = pywt.waverec2(coeffs, mode='periodization', wavelet=wavelet)
        rec_image = dequantize(rec_image)
        return rec_image


    class __EndDecoding(Exception):
        pass

    curr_i = dict(i=-1)
    def pop():
        i = curr_i['i']
        if i+1 >= len(d):
            print("END DECODING")
            raise __EndDecoding()
        curr_i['i'] = i + 1
        return d[curr_i['i']]


    ll_h = slices[0][1].stop
    ll_w = slices[0][2].stop
    lis = []
    for i in range(ll_h):
        for j in range(ll_w):
            if i % 2 == 0 and j% 2 == 0:
                continue
            lis.append(LisElement(i,j,'A'))
    lip = [(i,j) for i in range(ll_h) for j in range(ll_w)]
    lsp= []

    try:
        while n >=0:
            # sorting pass

            lsp_len = len(lsp)

            for i,j in lip:
                is_element_sig = pop()

                if is_element_sig:
                    # 1 or -1
                    signs = np.array([pop() for _ in range(c)]) * 2 - 1

                    values = np.array([pop() for _ in range(c)])
                    values = (values * 1.5) * 2**n

                    arr[:, i,j] = values * signs
                    lsp.append((i,j,))


            while len(lis) > 0:
                lis_element = lis.pop(0)

                i,j = lis_element.i, lis_element.j

                if lis_element.type == "A":
                    is_set_sig = pop()

                    if is_set_sig:
                        for k,l in get_offspring(lis_element.i, lis_element.j, h, w):
                            is_element_sig = pop()

                            if is_element_sig:
                                lsp.append((k,l,))

                                # either 1 or -1
                                signs = np.array([pop() for _ in range(c)]) * 2 - 1

                                # 0 or 1
                                values = np.array([pop() for _ in range(c)])

                                # 2^n <= abs(arr[k,l]) <= 2^(n+1)
                                values = (values * 1.5) * 2**n

                                arr[:, k,l] = arr[:, k, l] + values * signs
                            else:
                                lip.append((k,l))

                        has_descendents_past_offspring = i * 4 < h and j * 4 < w

                        if has_descendents_past_offspring:
                            lis.append(LisElement(i,j,"B"))
                                
                else:
                    # type B

                    descendents_past_offspring = sum([get_offspring(k,l,h,w) for (k,l) in get_offspring(i,j,h,w)], [])
                    is_l_significant = pop()
                    if is_l_significant:
                        for k,l in descendents_past_offspring:
                            lis.append(LisElement(k,l,"A"))


            # refinement pass
            print('refinement ', n)
            # does all the old elements, not added this round
            for lsp_i in range(lsp_len):
                i,j = lsp[lsp_i]
                bits = np.array([pop() for _ in range(c)])

                # either 1 or -1
                signs = (arr[:,i,j] > 0) * 2 - 1

                values = bits
                # either 0 or 1.5 * 2**n
                values = values * 2**n

                arr[:,i,j] = arr[:,i,j] + values * signs

            # quantization pass 
            n -= 1

    except __EndDecoding:
        return ret()

    return ret()

