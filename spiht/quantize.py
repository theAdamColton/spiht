import numpy as np

def quantize(arr, q_scale=10.):
    return (arr*q_scale).astype(np.int32)

def dequantize(arr, q_scale=10.):
    return arr / q_scale

