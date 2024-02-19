import torch
import imageio
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from spiht.spiht_wrapper import SpihtSettings, get_slices_and_h_w
from spiht.utils import imload
from spiht import encode_image,decode_image
from spiht.spiht import decode

spiht_settings = SpihtSettings(
       quantization_scale=75,
       color_model='IPT',
       per_channel_quant_scales=[8,1,1],
       wavelet="bior4.4",
       mode="symmetric",
)
level = 7

frames = 40

bpp_scale = 2
bpps = np.linspace(0.01, 0.7 ** (1/bpp_scale), frames) ** bpp_scale


print('loading image', sys.argv[1])
im = imload(sys.argv[1])

c,h,w = im.shape


max_bpp = bpps.max()
encoded = encode_image(
        im, spiht_settings,
        level,
        max_bits=int(max_bpp * h * w)
        )
original_bytes = encoded.encoded_bytes

font = ImageFont.truetype("./comic.ttf", 80)

ims = []
for bpp in tqdm(bpps):
    byte_len = int((bpp * h * w) / 8)
    byte_len = max(byte_len, 1)
    _bytes = original_bytes[:byte_len]

    encoded.encoded_bytes = _bytes
    rec_im = decode_image(
            encoded,
            spiht_settings
            )


    # decodes the coeffs manually
    slices, enc_h, enc_w = get_slices_and_h_w(h,w,spiht_settings,encoded.level)
    ll_h, ll_w = slices[0][1].stop, slices[0][2].stop
    dwt_coeffs_arr = decode(_bytes, encoded.max_n, c, enc_h, enc_w, ll_h, ll_w)
    dwt_coeffs_im = np.abs(dwt_coeffs_arr) * 75.0

    # rec_im might be smaller than dwt_coeffs_im
    dwt_coeffs_im = transforms.Resize((rec_im.shape[1], rec_im.shape[2]))(torch.from_numpy(dwt_coeffs_im)).numpy()

    out_im = np.concatenate(
            (
                rec_im,
                dwt_coeffs_im,
                ),
            -1)


    out_im = np.moveaxis(out_im, 0, -1)
    out_im = out_im.clip(0.0, 1.0)
    out_im = (out_im * 255).astype(np.uint8)
    out_im = Image.fromarray(out_im)
    ImageDraw.Draw(out_im).text((10, 10),f"BPP: {bpp:.4f}",(255,0,0),font=font)
    ims.append(np.array(out_im))

# duplicates last frame
for _ in range(5):
    ims.append(ims[-1])

print('saving animation...')

total_duration_ms = 15000
duration_per_frame_ms = total_duration_ms / len(ims)

w = imageio.get_writer('out.mp4',
                       #codec='hevc_vaapi',
                       format='FFMPEG', mode='I',
                       codec='h264',
                       fps = 1 / (duration_per_frame_ms / 1000))
for im in ims:
    w.append_data(im)
#imageio.mimsave("out.gif", ims, duration=duration_per_frame_ms, loop=0)
#print('optimizing gif size')
#optimize('out.gif')
