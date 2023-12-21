import unittest
import numpy as np
import matplotlib.pyplot as plt

from ..spiht import spiht_encode, spiht_decode
from ..utils import load_im, imshow, scale_0_1

class Tests(unittest.TestCase):
    def test_simple_encode(self):
        image = load_im("./images/lenna.png")
        d, n, h, w = spiht_encode(image)

    def test_simple_encode_decode(self):
        image = load_im("./images/lenna.png")
        d, n, h, w = spiht_encode(image)
        print("DECODING")
        image_hat = spiht_decode(d, n, h, w)
        imshow(scale_0_1(image_hat))
        imshow(image_hat)

