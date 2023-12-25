import math
from typing import List
import numpy as np
from pywt import wavedec2, waverec2
import pywt
from collections import namedtuple

from .utils import imshow, is_power_of_two, scale_0_1

def is_bit_set(x, n):
    return np.bitwise_and(np.abs(x), 2**n) > 0

def is_element_significant(x, n):
    return np.abs(x) >= 2**n

def has_descendents_past_offspring(i,j,h,w):
    if 2*i + 1 >= h or 2*j + 1 >= w:
        return False
    else:
        return True

def get_offspring(i,j,h,w,level):
    ll_h = h // 2 ** level
    ll_w = w // 2 ** level
    
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

def paint_descendents(i,j,h,w,level,out=None):
    if out is None:
        out = np.zeros((h,w))
    out[i,j] = out[i,j] + 1
    for k,l in get_offspring(i,j,h,w,level):
        paint_descendents(k,l,h,w,level,out)
    return out

def are_descendents_significant(arr, k, i, j, n, level):
    _, h, w =arr.shape

    for i,j in get_offspring(i,j,h,w,level):
        if is_set_significant(arr,k,i,j,n,level):
            return True

    return False

def is_set_significant(arr,k,i,j,n, level):
    _, h, w =arr.shape

    if is_element_significant(arr[k,i,j], n):
        return True

    for i,j in get_offspring(i,j,h,w, level):
        if is_set_significant(arr,k,i,j,n,level):
            return True

    return False

LisElement = namedtuple("LisElement", ['c', 'i', 'j', 'type'])

Q_SCALE = 10
def quantize(arr):
    return (arr*Q_SCALE).astype(np.int32)

def dequantize(arr):
    return arr / Q_SCALE

def spiht_encode(image, wavelet='bior4.4', level=7, max_bits=5000000):
    coeffs = wavedec2(image, mode='periodization', wavelet=wavelet, level=level)
    arr,slices = pywt.coeffs_to_array(coeffs, padding=None, axes=(-2,-1))
    arr = quantize(arr)

    coeffs = pywt.array_to_coeffs(dequantize(arr),slices, output_format='wavedec2')
    image = waverec2(coeffs, wavelet=wavelet, mode='periodization')

    c, h, w = arr.shape

    n = math.floor(math.log2(int(np.abs(arr).max())))
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
                    is_set_sig = are_descendents_significant(arr,k,i,j,n,level)
                    append_to_out(is_set_sig)

                    if is_set_sig:
                        # processes the four offspring
                        for offspring_i,offspring_j in get_offspring(i,j,h,w,level):
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
                    offspring = get_offspring(i,j,h,w,level)
                    descendents_past_offspring = []
                    for offspring_i, offspring_j in offspring:
                        descendents_past_offspring.extend(get_offspring(offspring_i,offspring_j,h,w,level))

                    is_l_significant = any(is_set_significant(arr,k,offspring_i,offspring_j,n,level) for (offspring_i,offspring_j) in descendents_past_offspring)
                    append_to_out(is_l_significant)
                    if is_l_significant:
                        for offspring_i, offspring_j in get_offspring(i,j,h,w,level):
                            lis.append(LisElement(k,offspring_i,offspring_j,"A"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)

            lis = lis_retain

            print('refinement')
            for lsp_i in range(lsp_len):
                k,i,j = lsp[lsp_i]
                bit = is_bit_set(arr[k,i,j], n)
                append_to_out(bit)

            # quantization pass 
            print(f'encoding quant pass n {n} kb:{curr_i["i"]/1000:.2f}')
            n -= 1

    except __EndEncoding:
        return dict(encoded=out, n=max_n, c=c, h=h, w=w, arr=arr, image=image)

    return dict(encoded=out, n=max_n, c=c, h=h, w=w, arr=arr, image=image)

def spiht_decode(d, n, h, w, c=3, wavelet='bior4.4', level=7, **kwargs):
    rec_arr = np.zeros((c,h,w), np.int32)

    # does this just to get the coeff slices
    dummy_image = rec_arr
    dummy_coeffs = wavedec2(dummy_image, mode='periodization', wavelet=wavelet, level=level)
    _, slices = pywt.coeffs_to_array(dummy_coeffs, padding=None, axes=(-2,-1))



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
                        for offspring_i,offspring_j in get_offspring(i,j,h,w,level):
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
                        for offspring_i, offspring_j in get_offspring(i,j,h,w,level):
                            lis.append(LisElement(k,offspring_i,offspring_j,"A"))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append(lis_element)

            lis = lis_retain

            # refinement pass
            print('refinement')

            for lsp_i in range(lsp_len):
                k,i,j = lsp[lsp_i]
                bit = pop()
                if bit:
                    rec_arr[k,i,j] = np.bitwise_or(rec_arr[k,i,j],2**n)
                else:
                    rec_arr[k,i,j] = np.bitwise_and(rec_arr[k,i,j], np.bitwise_not(np.uint32(2**n)))

            # quantization pass 
            print(f'encoding quant pass n {n} kb:{curr_i["i"]/1000:.2f}')
            n -= 1
    except __EndDecoding:
        coeffs = pywt.array_to_coeffs(dequantize(rec_arr), slices, output_format='wavedec2')
        rec_image = pywt.waverec2(coeffs, mode='periodization', wavelet=wavelet)
        return dict(rec_image=rec_image, coeffs=coeffs, rec_arr=rec_arr)


    coeffs = pywt.array_to_coeffs(dequantize(rec_arr), slices, output_format='wavedec2')
    rec_image = pywt.waverec2(coeffs, mode='periodization', wavelet=wavelet)
    return dict(rec_image=rec_image, coeffs=coeffs, rec_arr=rec_arr)

