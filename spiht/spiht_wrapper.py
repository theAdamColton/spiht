from dataclasses import asdict, dataclass
from typing import List, Optional, Any, Tuple, Union
import numpy as np
import pywt

from . import spiht as spiht_rs
from . import color_models

def quantize(arr, q_scale=10.):
    arr = arr * q_scale
    return arr.astype(np.int32)

def dequantize(arr, q_scale=10.):
    return arr / q_scale


ENCODER_DECODER_VERSION = "0.0.2"


@dataclass
class SpihtSettings:
    """
    These are parameters that are not particular to a single image,
    but instead pertain to the spiht algorithm.

    If these settings are pre-agreed upon, then they don't need to be saved when
    encoding images.

    The default settings are picked to work for general case 2D data.

    For RGB input images of natural scenes, I'd recommend the following settings:
        quantization_scale=1
        color_model ='ipt'
        per_channel_quant_scales=[100,20,20]

    Parameters:

    wavelet: type of pywt wavelet to use. The default is bior2.2 (also known as CDF 5/3).

    quantization_scale: the DWT coeffs are multiplied by this number before
        being encoded. The default value of 50 works with little perceptual loss
        for RGB pixels.
    
    color_model:
        the color_space (if any)
        used to encode the image. Supported color models a

    per_channel_quant_scales:
        This should be None, or a list of floats the same length as the number
        of channels in the image. This is useful for natural images encoded in
        color spaces such as IPT, because the data in the I channel is more
        important, and should have a higher quantization coefficient than the
        other two P and T channels.
    """
    wavelet: str = 'bior2.2'
    quantization_scale: float = 50.0
    mode: str = 'reflect'
    color_model: Optional[str] = None
    # This is an optional parameter that defines seperate quantization_scales
    # per channel
    # This is used for color spaces where some channels are more important than
    # others.
    per_channel_quant_scales: Optional[List[float]] = None

@dataclass
class EncodingResult:
    """
    encoded_bytes: bytes returned by the spiht encoder
    h: height of the original image
    w: width of the original image
    c: number of channels of the original image
    max_n: The starting n parameter used in the spiht encoder
    level: Optional number of DWT dec levels
    """
    encoded_bytes: bytes
    h: int
    w: int
    c: int
    max_n: int
    level: Optional[int]
    _encoding_version: str = ENCODER_DECODER_VERSION

    def to_dict(self):
        return {f"encoding_result{k}":v for k,v in asdict(self).items()}

    @staticmethod
    def from_dict(d):
        d = {k.removeprefix('encoding_result'):v for k,v in d.items() if k.startswith('encoding_result')}
        return EncodingResult(**d)


def get_slices_and_h_w(h: int, w: int, spiht_settings: SpihtSettings, level: Optional[int]):
    """
    Returns the same exact slices that would be used in the Wavedec
    same as pywt.coeffs_to_array slices

    Only works for a 3D array, with Wavedec2

    Returns:
    slices, height of rec array, width of rec array
    """
    shapes = pywt.wavedecn_shapes(
        (1, h, w),
        wavelet=spiht_settings.wavelet,
        mode=spiht_settings.mode,
        level=level,
        axes=(-2, -1),
    )
    *_, start_h, start_w = shapes[0]

    slices: List[Any] = [(slice(None), slice(start_h), slice(start_w))]
    for shape in shapes[1:]:
        shape_ad = shape["ad"]
        shape_da = shape["da"]
        shape_dd = shape["dd"]
        slices.append(
            {
                "ad": (
                    slice(None),
                    slice(0, shape_ad[1]),
                    slice(start_w, start_w + shape_ad[2]),
                ),
                "da": (
                    slice(None),
                    slice(start_h, start_h + shape_da[1]),
                    slice(0, shape_da[2]),
                ),
                "dd": (
                    slice(None),
                    slice(start_h, start_h + shape_dd[1]),
                    slice(start_w, start_w + shape_dd[2]),
                ),
            }
        )

        start_h += shape["dd"][1]
        start_w += shape["dd"][2]

    return slices, start_h, start_w


def encode_image(image: np.ndarray, spiht_settings:SpihtSettings=SpihtSettings(), level: Optional[int] = None, max_bits: Optional[int] = None):
    """
    Takes the DWT of the image, discretizes the DWT coeffs, and encodes it

    image: 3D ndarray of (C,H,W), containing floating point pixel values
    spiht_settings: Settings to be used for quantization and subsequent DWT transform
    level: integer number of DWT levels. 
    max_bits: max number of bits to use when encoding

    Returns EncodingResult
    """
    if image.ndim != 3:
        raise ValueError('image ndim must be 3: c,h,w')

    c,h,w = image.shape

    color_model = spiht_settings.color_model
    if color_model is not None:
        image = color_models.convert(image, 'RGB', color_model)


    coeffs = pywt.wavedec2(image, wavelet=spiht_settings.wavelet, level=level, mode=spiht_settings.mode)
    ll_h, ll_w = coeffs[0].shape[1], coeffs[0].shape[2]
    coeffs_arr,_ = pywt.coeffs_to_array(coeffs, axes=(-2,-1))

    per_channel_quant_scales = spiht_settings.per_channel_quant_scales
    if per_channel_quant_scales is not None:
        channel_mults = np.array(per_channel_quant_scales)
        coeffs_arr = channel_mults[:,None,None] * coeffs_arr

    coeffs_arr = quantize(coeffs_arr, spiht_settings.quantization_scale)

    if max_bits == None:
        # very large number
        max_bits = 99999999999999999

    encoded_bytes, max_n = spiht_rs.encode(coeffs_arr, ll_h, ll_w, max_bits)

    encoding_result = EncodingResult(
            encoded_bytes,
            h,
            w,
            c,
            max_n,
            level,
        )
    
    return encoding_result


def decode_image(encoding_result: EncodingResult, spiht_settings: SpihtSettings, return_metadata:bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Decodes the encoding_result to pixel values

    encoding_result: EncodingResult
        This is the encoding_result that was returned by the encode_image
        function

    spiht_settings: SpihtSettings
        These are the settings passed to the encode_image function

    return_metadata: Bool
        If true, will return a tuple of reconstructed image and spiht metadata
        The spiht metadata is a 2D numpy i32 array containing at each position
        a vector describing the internal state of the spiht decoder. The number
        of rows is equal to the number of encoded bits.
    """
    d = decode_rec_array(encoding_result, spiht_settings, return_metadata)
    spiht_metadata = d.pop("spiht_metadata", None)
    image = decode_from_rec_arr(**d, spiht_settings=spiht_settings)

    if return_metadata:
        return image, spiht_metadata
    else:
        return image

def decode_rec_array(encoding_result: EncodingResult, spiht_settings: SpihtSettings, return_metadata:bool=False):
    encoded_bytes = encoding_result.encoded_bytes
    h = encoding_result.h
    w = encoding_result.w
    c = encoding_result.c
    max_n = encoding_result.max_n
    level = encoding_result.level

    if encoding_result._encoding_version != ENCODER_DECODER_VERSION:
        raise ValueError(encoding_result._encoding_version)

    slices, enc_h, enc_w = get_slices_and_h_w(h,w,spiht_settings,level)
    ll_h, ll_w = slices[0][1].stop, slices[0][2].stop

    if return_metadata:
        top_slice = [
                (slices[0][1].start or 0, slices[0][1].stop),
                (slices[0][2].start or 0, slices[0][2].stop),
                ]
        other_slices = []
        for slice_level in slices[1:]:
            slice_filters = []
            for filter_key in ["da", "ad", "dd"]:
                slice_filter = slice_level[filter_key]
                slice_filters.append(
                        [
                            (slice_filter[1].start, slice_filter[1].stop),
                            (slice_filter[2].start, slice_filter[2].stop),
                        ]
                    )
            other_slices.append(slice_filters)

        rec_arr, spiht_metadata = spiht_rs.decode_with_metadata(encoded_bytes, max_n, c, enc_h, enc_w, ll_h, ll_w, top_slice, other_slices)
    else:
        rec_arr = spiht_rs.decode(encoded_bytes, max_n, c, enc_h, enc_w, ll_h, ll_w)
        spiht_metadata = None

    return dict(
            rec_arr = rec_arr, slices=slices, spiht_metadata=spiht_metadata, h=h, w=w, level=level
            )

def decode_from_rec_arr(rec_arr:np.ndarray, h:int, w:int, level, spiht_settings:SpihtSettings, slices=None):
    wavelet = spiht_settings.wavelet
    quantization_scale = spiht_settings.quantization_scale
    mode=spiht_settings.mode
    color_model=spiht_settings.color_model
    per_channel_quant_scales=spiht_settings.per_channel_quant_scales

    if slices is None:
        slices, _, _ = get_slices_and_h_w(h,w,spiht_settings,level)


    if per_channel_quant_scales is not None:
        channel_mults = np.array(per_channel_quant_scales)
        rec_arr = rec_arr / channel_mults[:,None,None]

    rec_arr = dequantize(rec_arr, quantization_scale)
    rec_coeffs = pywt.array_to_coeffs(rec_arr, slices, output_format='wavedec2')
    rec_image = pywt.waverec2(rec_coeffs, wavelet, mode=mode)

    if color_model is not None:
        rec_image = color_models.convert(rec_image, color_model, "RGB")

    return rec_image
