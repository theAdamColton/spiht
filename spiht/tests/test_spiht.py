import unittest
import os
from .. import encode_image, decode_image
from ..utils import load_im, imshow

class Tests(unittest.TestCase):
    def test_encode_decode_images(self):
        for image_file in os.listdir("./images/"):
            image = load_im(f"./images/{image_file}")
            _,h,w = image.shape
            encoded = encode_image(image, mode='periodization', wavelet='bior4.4', level=4, max_bits =h*w)
            decoded_image = decode_image(encoded)
            # Uncomment the following line to get the script to show the two images
            #imshow(decoded_image)
