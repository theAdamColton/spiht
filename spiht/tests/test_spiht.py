import unittest
import os
from .. import encode_image, decode_image
from ..utils import imload, imshow

class Tests(unittest.TestCase):
    def test_encode_decode_images(self):
        for image_file in os.listdir("./images/"):
            image = imload(f"./images/{image_file}")
            encoded = encode_image(image)
            decoded_image = decode_image(encoded)
            # Uncomment the following line to get the script to show the two images
            #imshow(decoded_image)
