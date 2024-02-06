import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import reduce, rearrange

def bytes_to_bits(spiht_bytes: bytes):
    np_bytes = np.frombuffer(spiht_bytes, np.uint8)
    np_bits = np.unpackbits(np_bytes, bitorder='little')
    return np_bits


def imload(path) -> np.ndarray:
    im= np.asarray(Image.open(path))
    if im.ndim > 2:
        im = np.moveaxis(im, -1, 0)
    else:
        im = im[None,:,:]

    im = im / 255
    return im

def scale_0_1(x):
    x = rearrange(x, '... h w -> h w ...')
    _min = reduce(x, 'h w ... -> ...', 'min')
    _max = reduce(x, 'h w ... -> ...', 'max')
    x = (x - _min) / (_max - _min)
    x = rearrange(x, 'h w ... -> ... h w')
    return x

def imshow(x, ax=None, scale=False):
    if x.ndim > 2:
        x = np.moveaxis(x,0,-1)
    if scale:
        x = scale_0_1(x)

    if ax is None:
        ax = plt
        ax.imshow(x)
        ax.axis("off")
        plt.show()
    else:
        ax.axis("off")
        ax.tick_params(
            axis="both",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        ax.imshow(x)


