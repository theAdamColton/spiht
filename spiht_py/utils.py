import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import reduce, rearrange

def load_im(path) -> np.ndarray:
    im= np.asarray(Image.open(path))
    return np.moveaxis(im, -1, 0)

def scale_0_1(x):
    x = rearrange(x, '... h w -> h w ...')
    _min = reduce(x, 'h w ... -> ...', 'min')
    _max = reduce(x, 'h w ... -> ...', 'max')
    x = (x - _min) / (_max - _min)
    x = rearrange(x, 'h w ... -> ... h w')
    return x

def is_power_of_two(target: int) -> int:
    if target > 1:
        for i in range(1, int(target)):
            if 2**i >= target:
                return 2**i
    return 1

def imshow(x, ax=None, scale=False):
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


