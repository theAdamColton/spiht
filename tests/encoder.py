import unittest
import numpy as np

from spiht_py.spiht import spiht_encode
from spiht_py.utils import load_im

class EncoderTests(unittest.TestCase):
    def test_simple_encode(self):
        image = load_im("./images/lenna.png")
        spiht_encode(image)


