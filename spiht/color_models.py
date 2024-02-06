import numpy as np
import colour

SUPPORTED_MODELS = set(colour.COLOURSPACE_MODELS)

def convert(im, src, dest):
    if src not in SUPPORTED_MODELS:
        raise ValueError(f'{src} is not a supported color model. Supported models are {SUPPORTED_MODELS}')
    if dest not in SUPPORTED_MODELS:
        raise ValueError(f'{dest} is not a supported color model. Supported models are {SUPPORTED_MODELS}')
    im = np.moveaxis(im, 0, -1)
    im = colour.convert(im, src, dest)
    return np.moveaxis(im, -1, 0)
