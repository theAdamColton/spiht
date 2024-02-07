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
