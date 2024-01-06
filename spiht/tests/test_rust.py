import unittest
import matplotlib.pyplot as plt
import numpy as np
import pywt

from ..utils import load_im, imshow, scale_0_1
from ..quantize import quantize, dequantize
from .. import spiht as spiht_rs

class RustTests(unittest.TestCase):
    def test_encode_decode(self):
        image = load_im("./images/skiing.jpg")
        level=5
        q_scale = 50
        wavelet = 'bior4.4'
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode='periodization')
        coeffs_arr,slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1),)
        coeffs_arr = quantize(coeffs_arr, q_scale)
        ll_h, ll_w = coeffs[0].shape[1], coeffs[0].shape[2]
        data, max_n = spiht_rs.encode(coeffs_arr, ll_h, ll_w, 999999999999)

        c,h,w = coeffs_arr.shape
        rec_arr = spiht_rs.decode(data, max_n, c,h,w, ll_h, ll_w)
        rec_arr_dequant = dequantize(rec_arr, q_scale)
        rec_coeffs = pywt.array_to_coeffs(rec_arr_dequant, slices, output_format='wavedec2')
        rec_image = pywt.waverec2(rec_coeffs, wavelet, mode='periodization')

        print('image encoded to {} kb', len(data) / 1024)

        f,ax = plt.subplots(2)
        imshow(image, ax=ax[0])
        imshow(rec_image, ax=ax[1])
        plt.show()
        plt.close()

        f,ax = plt.subplots(2)
        imshow(rec_arr*1000, ax=ax[0])
        imshow(coeffs_arr*1000, ax=ax[1])
        plt.show()
        plt.close()

        self.assertTrue(np.array_equal(coeffs_arr, rec_arr))
        
