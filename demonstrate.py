import os
import numpy as np
import matplotlib.pyplot as plt
from einops import einsum

from spiht.utils import load_im,imshow
from spiht import encode_image, decode_image
from spiht.spiht_py import spiht_encode, spiht_decode

# some helper code to convert back and forth between RGB and IPT
# https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

# sRGB -> XYZ D65
MsRGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
# XYZ D65 -> LMS
MHPE = np.array(
    [[0.4002, 0.7076, -0.0807], [-0.2280, 1.1500, 0.0612], [0, 0, 0.9184]]
)

# LMS -> ipt
Mipt = np.array(
    [[0.4, 0.4, 0.2], [4.455, -4.851, 0.3960], [0.8056, 0.3572, -1.1628]]
)
Trgb2lms = MHPE @ MsRGB
Tlms2rgb = np.linalg.inv(Trgb2lms)


def channel_mult(M, x):
    return einsum(M, x, "i j, ... j h w -> ... i h w")

def rgb_to_lms(x):
    return channel_mult(
        Trgb2lms,
        x,
    )

def lms_to_rgb(x):
    return channel_mult(
        Tlms2rgb,
        x,
    )

def rgb_to_ipt(x):
    """
    page 147
    https://scholarworks.rit.edu/theses/2858/
    """
    x = rgb_to_lms(x)
    mask = x >= 0.0
    x[mask] = x[mask] ** 0.43
    x[~mask] = -1 * (-1 * x[~mask]) ** 0.43
    return channel_mult(
        Mipt,
        x,
    )

def ipt_to_rgb(x):
    """
    page 147
    https://scholarworks.rit.edu/theses/2858/
    """
    x = channel_mult(
        np.linalg.inv(Mipt),
        x,
    )
    mask = x >= 0.0
    x[mask] = x[mask] ** 2.3256
    x[~mask] = -1 * (-1 * x[~mask]) ** 2.3256
    return lms_to_rgb(x)


for image_file in os.listdir("./images/"):
    print(image_file)
    image = load_im(f"./images/{image_file}")
    image = rgb_to_ipt(image)
    _,h,w = image.shape
    wavelet='bior4.4'
    level=5
    quantization_scale=10
    mode='antireflect'

    bpps = [0.1, 0.5, 1]
    f,axg=plt.subplots(len(bpps),2)
    for bpp, ax in zip(bpps, axg):
        max_bits = int(h*w*bpp)
        encoded = encode_image(image, mode=mode, level=level, wavelet=wavelet, max_bits=max_bits)
        decoded_image = decode_image(encoded)
        
        encoded_py = spiht_encode(image, level=level, wavelet=wavelet, max_bits=max_bits, quantization_scale=quantization_scale)
        decoded_py = spiht_decode(level=level, wavelet=wavelet, quantization_scale=quantization_scale, **encoded_py)
        decoded_image_py = decoded_py['rec_image']

        decoded_image = ipt_to_rgb(decoded_image)
        decoded_image_py = ipt_to_rgb(decoded_image_py)

        imshow(decoded_image,ax[0])
        ax[0].set_title(f'{len(encoded.encoded_bytes)/1024:.2f}kb  bpp{bpp}')
        imshow(decoded_image_py, ax[1])
        ax[1].set_title(f'{(len(encoded_py["d"])/8)/1024:.2f}kb bpp{bpp}')
    plt.subplots_adjust(bottom=0, hspace=0, wspace=0,left=0, right=1, top=0.95)
    plt.show()
