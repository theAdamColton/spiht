"""
This file provides a reference implementation of the spiht encoder and decoder

The python encoder is very slow and not practical for any real use.

The python decoder is not noticably slow.
"""

import math
import numpy as np
from pywt import wavedec2, waverec2
import pywt
from collections import namedtuple

from .spiht_wrapper import EncodingResult

def set_bit(x,n,bit):
    sign = x>=0
    if bit:
        if sign:
            return x | (1 << n)
        else:
            return -((-x) | 1 <<n)
    else:
        if sign:
            return x & ~(1<<n)
        else:
            return -((-x) & ~(1<<n))

def is_bit_set(x, n):
    return np.bitwise_and(np.abs(x), 2**n) > 0

def is_element_significant(x, n):
    return np.abs(x) >= 2**n

def has_descendents_past_offspring(i,j,h,w):
    if 2*i + 1 >= h or 2*j + 1 >= w:
        return False
    else:
        return True

def get_offspring(i,j,h,w,ll_h, ll_w):
    if i < ll_h and j < ll_w:
        if i%2 == 0 and j%2 == 0:
            return []
        # index relative to the top left chunk corner
        # can be (0,0), (0,2), (2,0), (2,2)
        sub_child_i, sub_child_j = i // 2 * 2, j//2 * 2
        # can be (0,1), (1,0) or (1,1)
        chunk_i, chunk_j = i%2, j%2
        return [
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j + 1),
                (chunk_i * ll_h + sub_child_i+1, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i + 1, chunk_j * ll_w + sub_child_j + 1),
                ]

    if 2*i+1 >= h or 2*j+1 >= w:
        return []

    return [
            (2*i,2*j), 
            (2*i, 2*j+1),
            (2*i+1, 2*j),
            (2*i+1, 2*j+1),
        ]

def are_descendents_significant(arr, k, i, j, n, ll_h,ll_w):
    _, h, w =arr.shape

    for i,j in get_offspring(i,j,h,w,ll_h,ll_w):
        if is_set_significant(arr,k,i,j,n,ll_h,ll_w):
            return True

    return False

def is_set_significant(arr,k,i,j,n,ll_h,ll_w):
    _, h, w =arr.shape

    if is_element_significant(arr[k,i,j], n):
        return True

    for l,m in get_offspring(i,j,h,w,ll_h,ll_w):
        if is_set_significant(arr,k,l,m,n,ll_h,ll_w):
            return True

    return False

def quantize(arr, quantization_scale=10.):
    return (arr*quantization_scale).astype(np.int32)

def dequantize(arr, quantization_scale=10.):
    return arr / quantization_scale

LisElement = namedtuple("LisElement", ['c', 'i', 'j', 'type'])

class EndDecoding(Exception):
    pass

class EndEncoding(Exception):
    pass

def encode_image_py(image: np.ndarray, wavelet='bior4.4', level=6, max_bits=None, quantization_scale=50, mode='periodization') -> EncodingResult:
    if image.ndim != 3:
        raise ValueError('image ndim must be 3: c,h,w')

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode=mode)
    ll_h, ll_w = coeffs[0].shape[1], coeffs[0].shape[2]
    arr,slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1))

    arr = quantize(arr, quantization_scale)

    c,h,w = arr.shape

    if max_bits == None:
        # very large number
        max_bits = 99999999999999999

    n = math.floor(math.log2(int(np.abs(arr).max())))
    max_n = n


    curr_i = dict(i=0)
    out = [0] * max_bits
    def append_to_out(*x):
        i = curr_i['i']
        for y in x:
            if i >= max_bits:
                raise EndEncoding()
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
            for k in range(c):
                lis.append(LisElement(k,i,j,'A'))
    lip = []
    for i in range(ll_h):
        for j in range(ll_w):
            for k in range(c):
                lip.append((k,i,j,))
    lsp= []


    try:
        while n >=0:
            # sorting pass

            # stores the lsp len at the beginning of this n iteration
            lsp_len = len(lsp)

            new_lip = []
            for k,i,j in lip:
                is_element_sig = is_element_significant(arr[k, i, j], n)
                append_to_out(is_element_sig)

                if is_element_sig:
                    append_to_out(arr[k,i,j]>=0)
                    lsp.append((k,i,j))

                else:
                    new_lip.append((k,i,j))
            lip = new_lip

            lis_retain = []

            while len(lis) > 0:
                lis_element = lis.pop(0)

                k,i,j = lis_element.c, lis_element.i, lis_element.j

                if lis_element.type == "A":
                    is_set_sig = are_descendents_significant(arr,k,i,j,n,ll_h,ll_w)
                    append_to_out(is_set_sig)

                    if is_set_sig:
                        # processes the four offspring
                        for offspring_i,offspring_j in get_offspring(i,j,h,w,ll_h,ll_w):
                            is_element_sig = is_element_significant(arr[k, offspring_i, offspring_j], n)
                            append_to_out(is_element_sig)

                            if is_element_sig:
                                lsp.append((k,offspring_i,offspring_j))

                                append_to_out(arr[k,offspring_i,offspring_j] >= 0)
                            else:
                                lip.append((k, offspring_i, offspring_j))

                        l_exists = has_descendents_past_offspring(i,j,h,w)
                        if l_exists:
                            lis.append(LisElement(k,i,j,"B"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)
                                
                else:
                    # type B
                    offspring = get_offspring(i,j,h,w,ll_h, ll_w)
                    descendents_past_offspring = []
                    for offspring_i, offspring_j in offspring:
                        descendents_past_offspring.extend(get_offspring(offspring_i,offspring_j,h,w,ll_h,ll_w))

                    is_l_significant = any(is_set_significant(arr,k,offspring_i,offspring_j,n,ll_h,ll_w) for (offspring_i,offspring_j) in descendents_past_offspring)
                    append_to_out(is_l_significant)
                    if is_l_significant:
                        for offspring_i, offspring_j in get_offspring(i,j,h,w,ll_h,ll_w):
                            lis.append(LisElement(k,offspring_i,offspring_j,"A"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)

            lis = lis_retain

            #print('refinement')
            for lsp_i in range(lsp_len):
                k,i,j = lsp[lsp_i]
                bit = is_bit_set(arr[k,i,j], n)
                append_to_out(bit)

            # quantization pass 
            print(f'encoding quant pass n {n} kb:{(curr_i["i"]/8)/1024:.2f}')
            n -= 1

    except EndEncoding:
        pass

    encoding_result = EncodingResult(
            out,
            h,
            w,
            c,
            max_n,
            ll_h,
            ll_w,
            wavelet,
            quantization_scale,
            slices,
            mode,
            )

    return encoding_result

def decode_image_py(encoding_result: EncodingResult) -> np.ndarray:
    d = encoding_result.encoded_bytes
    h = encoding_result.h
    w = encoding_result.w
    c = encoding_result.c
    n = encoding_result.max_n
    ll_h = encoding_result.ll_h
    ll_w = encoding_result.ll_w
    wavelet = encoding_result.wavelet
    quantization_scale = encoding_result.quantization_scale
    slices=encoding_result.slices
    mode=encoding_result.mode

    # does a dummy encoding just to get the coeff slices
    rec_arr = np.zeros((c,h,w), dtype=np.int32)


    curr_i = dict(i=-1)
    def pop():
        i = curr_i['i']
        if i+1 >= len(d):
            raise EndDecoding()
        curr_i['i'] = i + 1
        return d[curr_i['i']]

    ll_h = slices[0][1].stop
    ll_w = slices[0][2].stop
    lis = []
    for i in range(ll_h):
        for j in range(ll_w):
            if i % 2 == 0 and j% 2 == 0:
                continue
            for k in range(c):
                lis.append(LisElement(k,i,j,'A'))
    lip = []
    for i in range(ll_h):
        for j in range(ll_w):
            for k in range(c):
                lip.append((k,i,j,))
    lsp= []


    try:
        while n >= 0:
            # sorting pass
            # stores the lsp len at the beginning of this n iteration
            lsp_len = len(lsp)

            new_lip = []
            for k,i,j in lip:
                is_element_sig = pop()

                if is_element_sig:
                    # 1 or -1
                    sign = pop() * 2 - 1

                    rec_arr[k,i,j] = 1.5 * 2**n * sign
                    lsp.append((k,i,j,))
                else:
                    new_lip.append((k,i,j))
            lip=new_lip


            lis_retain = []

            while len(lis) > 0:
                lis_element = lis.pop(0)

                k,i,j = lis_element.c, lis_element.i, lis_element.j

                if lis_element.type == "A":
                    is_set_sig = pop()

                    if is_set_sig:
                        # processes the four offspring
                        for offspring_i,offspring_j in get_offspring(i,j,h,w,ll_h,ll_w):
                            is_element_sig = pop()

                            if is_element_sig:
                                lsp.append((k,offspring_i,offspring_j))
                                # either 1 or -1
                                sign = pop() * 2 - 1

                                rec_arr[k,offspring_i,offspring_j] = 1.5*2**n * sign
                            else:
                                lip.append((k, offspring_i, offspring_j))

                        l_exists = has_descendents_past_offspring(i,j,h,w)
                        if l_exists:
                            lis.append(LisElement(k,i,j,"B"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)
                                
                else:
                    # type B
                    is_l_significant = pop()
                    if is_l_significant:
                        for offspring_i, offspring_j in get_offspring(i,j,h,w,ll_h,ll_w):
                            lis.append(LisElement(k,offspring_i,offspring_j,"A"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)

            lis = lis_retain

            # refinement pass
            for lsp_i in range(lsp_len):
                k,i,j = lsp[lsp_i]

                bit = pop()
                rec_arr[k,i,j] = set_bit(rec_arr[k,i,j], n, bit)

            # quantization pass 
            print(f'decoding quant pass n {n} kb:{(curr_i["i"]/8) / 1024:.2f}kb')
            n -= 1

    except EndDecoding:
        pass


    coeffs = pywt.array_to_coeffs(dequantize(rec_arr, quantization_scale), slices, output_format='wavedec2')
    rec_image = pywt.waverec2(coeffs, mode=mode, wavelet=wavelet)
    return rec_image

