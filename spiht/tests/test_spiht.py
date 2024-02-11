import numpy as np
import unittest
import os

from spiht.spiht_wrapper import SpihtSettings
from .. import encode_image, decode_image
from ..utils import imload, imshow

class Tests(unittest.TestCase):
    def test_encode_decode_images(self):
        for image_file in os.listdir("./images/"):
            image = imload(f"./images/{image_file}")
            spiht_settings = SpihtSettings()
            encoded = encode_image(image, spiht_settings=spiht_settings)
            decoded_image = decode_image(encoded, spiht_settings)
            # Uncomment the following line to get the script to show the two images
            #imshow(decoded_image)

    def test_encode_decode_with_metadata(self):
        for image_file in os.listdir("./images/"):
            image = imload(f"./images/{image_file}")
            spiht_settings = SpihtSettings()
            encoded = encode_image(image, spiht_settings=spiht_settings)
            decoded_image, spiht_metadata = decode_image(encoded, spiht_settings, return_metadata=True)
            decoded_image_2 = decode_image(encoded, spiht_settings, return_metadata=False)
            self.assertTrue(np.allclose(decoded_image, decoded_image_2))
            # Uncomment the following line to get the script to show the two images
            #imshow(decoded_image)
