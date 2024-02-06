"""
This script uses the rust, and python implementations of spiht to encode and
decode image pixels. Image pixels are represented using the IPT color space,
instead of RGB. The resulting decoded images are shown using matplotlib.
"""


import os
import matplotlib.pyplot as plt
from spiht import SpihtSettings

from spiht.utils import imload,imshow
from spiht import encode_image, decode_image


# main script
wavelet='bior2.2'
level=None
quantization_scale=1.
mode='reflect'
color_space='ipt'
per_channel_quant_scales=[100.,20.,20.]
spiht_settings = SpihtSettings(
        wavelet,
        quantization_scale,
        mode,
        color_space,
        per_channel_quant_scales
        )

# bits per pixel
bpps = [0.1, 0.5, 1.0]

for image_file in os.listdir("./images/"):
    image_file = './images/' + image_file
    print(image_file)
    image = imload(image_file)

    _,h,w = image.shape

    # currently, pywt only supports even resolutions
    if h%2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1
    image = image[:,:h,:w]

    f,axg=plt.subplots(len(bpps), dpi=300, figsize=(20,10))
    for bpp, ax in zip(bpps, axg):
        max_bits = int(h*w*bpp)


        # First, encodes and decodes using the rust implementation
        # This is the recommended way to encode and decode images
        encoded = encode_image(image,spiht_settings,level,max_bits)
        decoded_image = decode_image(encoded)

        imshow(decoded_image,ax)
        #ax.set_title(f'decoded {len(encoded.encoded_bytes)/1024:.2f}kb  bpp{bpp}')

    plt.subplots_adjust(bottom=0, hspace=0, wspace=0,left=0, right=1, top=0.95)
    plt.show()

    input("ctrl c to exit, enter to continue")
