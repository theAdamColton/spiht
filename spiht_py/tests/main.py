import unittest
import numpy as np
import matplotlib.pyplot as plt

from ..spiht import spiht_encode, spiht_decode, get_offspring
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

    def test_simple_encode(self):
        image = load_im("./images/lenna.png")
        result = spiht_encode(image)

    def test_simple_encode_decode(self):
        image = load_im("./images/lenna.png")
        result = spiht_encode(image)
        print("DECODING")
        decoding_result = spiht_decode(result['encoded'], **result)
        rec_image = decoding_result['rec_image']
        imshow(scale_0_1(rec_image))
        imshow(rec_image)
