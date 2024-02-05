"""
encode/decode a single image using spiht

saves an image of the decoded image
"""
import math
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from spiht.spiht_wrapper import SpihtSettings, get_slices_and_h_w
import time

from spiht.utils import load_im
from spiht import encode_image,decode_image

parser =ArgumentParser()
parser.add_argument('image_filename')
parser.add_argument('--bpp', help='bits per pixel', type=float, default=0.1)
parser.add_argument('--quantization_scale', default=1.0, type=float)
parser.add_argument('--level', help='wavedec2 level. default is set so that the highest DWT level has a width and height of 4.', default=None, type=int)
parser.add_argument('--wavelet', help='wavedec2 wavelet', default='bior2.2', type=str)
parser.add_argument('--mode', help='wavedec2 mode', default='reflect', type=str)
parser.add_argument('--disable_ipt', action="store_true")
parser.add_argument('--out', help='save reconstructed image to this file path', type=str, default='reconstructed.png')

def main(args):
    im = load_im(args.image_filename)

    c,h,w = im.shape

    if args.level is None:
        level = min(
                math.log2(h/8),
                math.log2(w/8)
                )
        level = math.floor(level)
    else:
        level=args.level

    pixels = h*w
    max_bits = round(args.bpp * pixels)

    spiht_settings = SpihtSettings(
           quantization_scale=args.quantization_scale,
           mode=args.mode,
           wavelet=args.wavelet,
           color_space=None if args.disable_ipt else 'ipt',
           per_channel_quant_scales=None if args.disable_ipt else [50,15,15],
            )
    print(f"Starting encoding of image {c} {h} {w}")
    st = time.time()
    encoded = encode_image(
           im,
           spiht_settings,
           level,
           max_bits,
    )
    et = time.time()
    print(f"Encoding done in {et-st:.3f}s. Image encoded to {len(encoded.encoded_bytes) / 1024:.2f}kb")
    print(f"   levels: {encoded.level}")
    print(f"    max n: {encoded.max_n}")
    slices, enc_h, enc_w = get_slices_and_h_w(h,w,spiht_settings,encoded.level)
    ll_h, ll_w = slices[0][1].stop, slices[0][2].stop
    print(f"ll_h ll_w: {ll_h, ll_w}")
    st = time.time()
    dec_im = decode_image(encoded)
    et = time.time()
    print(f"Decoding done in {et-st:.3f}s. L2 distance: {((im-dec_im)**2).mean():.5f}")

    if c==1:
        dec_im = dec_im[0]
    else:
        dec_im = np.moveaxis(dec_im, 0, -1)

    dec_im = dec_im.clip(0.0, 1.0)
    dec_im = (dec_im * 255).astype(np.uint8)

    dec_im_pil = Image.fromarray(dec_im)
    dec_im_pil.save(args.out)

    print("Saved to ", args.out)


if __name__ == "__main__":
    args=parser.parse_args()
    main(args)
