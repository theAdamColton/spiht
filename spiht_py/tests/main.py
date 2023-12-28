import unittest
import numpy as np
import matplotlib.pyplot as plt
import pywt

from ..spiht import spiht_encode, spiht_decode, get_offspring, dequantize, quantize, simple_spiht_encode, simple_spiht_decode
from ..utils import load_im, imshow, scale_0_1

class Tests(unittest.TestCase):
    def test_get_ll_offspring(self):
        h,w=512,512
        level=8
        # ll_h = 4
        # ll_w = 4
        offspring=get_offspring(0,0,h,w,level)
        self.assertTrue(len(offspring)==0)
        offspring=get_offspring(0,1,h,w,level)

    def test_encode(self):
        image = load_im("./images/lenna.png")
        result = simple_spiht_encode(image)

    def test_encode_decode(self):
        image = load_im("./images/lenna.png")
        result = spiht_encode(image)
        print("DECODING")
        decoding_result = spiht_decode(result['encoded'], **result)
        rec_image = decoding_result['rec_image']
        arr=result['arr']
        rec_arr=decoding_result['rec_arr']

        f,ax = plt.subplots(2)
        imshow(arr*1e10, ax=ax[0])
        imshow(rec_arr*1e10, ax=ax[1])
        plt.show()
        plt.close()


        f,ax = plt.subplots(2)
        imshow(result['image'], ax=ax[0])
        imshow(rec_image, ax=ax[1])
        plt.show()
        plt.close()

        diff = np.abs(arr - rec_arr)
        imshow(diff * 1000)
        plt.show()


    def test_simple_encode_decode(self):
        image = load_im("./images/lenna.png")
        result = simple_spiht_encode(image)
        print("DECODING")
        decoding_result = simple_spiht_decode(result['encoded'], **result)
        rec_image = decoding_result['rec_image']
        arr=result['arr']
        rec_arr=decoding_result['rec_arr']

        f,ax = plt.subplots(2)
        imshow(arr*1e10, ax=ax[0])
        imshow(rec_arr*1e10, ax=ax[1])
        plt.show()
        plt.close()


        f,ax = plt.subplots(2)
        imshow(result['image'], ax=ax[0])
        imshow(rec_image, ax=ax[1])
        plt.show()
        plt.close()

        diff = np.abs(arr - rec_arr)
        imshow(diff * 1000)
        plt.show()


