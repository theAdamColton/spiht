import unittest
import matplotlib.pyplot as plt
import numpy as np

from .utils import load_im, imshow, scale_0_1
import pywt
import spiht_rs
from .spiht import quantize, dequantize

class RustTests(unittest.TestCase):
    def test_encode_decode(self):
        image = load_im("./images/lenna.png")
        c,h,w = image.shape
        level=8
        q_scale = 50
        wavelet = 'bior4.4'
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode='periodization')
        coeffs_arr,slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1), padding=None)
        coeffs_arr = quantize(coeffs_arr, q_scale)
        ll_h, ll_w = coeffs[0].shape[1], coeffs[0].shape[2]
        data, max_n = spiht_rs.encode_spiht(coeffs_arr, ll_h, ll_w, 999999999999)
        rec_arr = spiht_rs.decode_spiht(data, max_n, c,h,w, ll_h, ll_w)
        rec_arr = dequantize(rec_arr, q_scale)
        rec_coeffs = pywt.array_to_coeffs(rec_arr, slices, output_format='wavedec2')
        rec_image = pywt.waverec2(rec_coeffs, wavelet, mode='periodization')

        print('image encoded to {} kb', (len(data) / 8) / 1024)

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
        

