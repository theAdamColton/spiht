import unittest
import numpy as np
import pywt

from .. import encode_image, decode_image
from ..quantize import quantize, dequantize

class Tests(unittest.TestCase):
    def test_encode_decode_random(self):
        rng = np.random.default_rng(42)
        quantization_scale=50
        wavelet='bior4.4'
        level=6

        for _ in range(20):
            c = rng.integers(1,8)
            h = rng.integers(128,1024)
            w = rng.integers(128,1024)
            image = rng.standard_normal((c,h,w))

            coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode='periodization')
            coeffs_arr,slices = pywt.coeffs_to_array(coeffs, axes=(-2,-1), padding=0)
            coeffs_arr= quantize(coeffs_arr, quantization_scale)
            coeffs_arr = dequantize(coeffs_arr, quantization_scale)
            coeffs = pywt.array_to_coeffs(coeffs_arr, slices, output_format='wavedec2')
            image_ground_truth = pywt.waverec2(coeffs, wavelet, mode='periodization')

            encoding_result = encode_image(image)
            decoded_image = decode_image(encoding_result)

            # currently does not pass
            self.assertTrue(np.allclose(image_ground_truth, decoded_image, 1e-4))
        
