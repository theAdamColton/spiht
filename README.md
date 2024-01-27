# spiht-py

<p align="center">
<img width="220" alt="Screenshot 2024-01-27 at 11 15 47" src="https://github.com/theAdamColton/spiht-py/assets/72479734/52d8375c-7ed9-44c4-9ace-71edd14dc25a">
</p>

A python/numpy/rust implementation of the [spiht](https://spiht.com/spiht1.html) algorithm. (Set Partitioning in Hierarchical Trees)

SPIHT is an algorithm originally concieved for image compression. It uses the fact that when a natural image is transformed into its DWT coefficients, the coefficients are highly correlated in a organized hierarchical ordering. It also by default does not use any blocking/tiling like in JPEG, which can help reduce artifacts. 

# Installation and Usage

You will need the `maturin` CLI tool which is used to install hybrid Rust/python projects. Once `maturin` is install you can run `maturin develop --release`. This command will install spiht as a module in the current python virtualenv. For an example as to how to import and use spiht, checkout `demonstrate.py`.

# Description

I started by writing a naive python/numpy implementation. This turned out to be very slow. The bottleneck in the python code was the encoder. The encoder has to determine whether a DWT coefficient is significant or not by recursively checking the significance of all of its descendants. In native python, doing this recursive search gets really slow without any specific optimizations.

I wrote a version in Rust without any performance hacks that performes adequately. The original native python version could take around 3-5 seconds to encode a large image. The Rust version does this almost in imperceptable time. 

The standalone native python encoder/decoder can be found in `spiht/spiht_py.py`. The core rust code can be found in `src/encoder_decoder.rs`

# Extra features

You can convert RGB pixels into the IPT color space. Compressing images in this color space better preserves details. Additionally, quantization scales can be specified by channel. Using IPT and a [50, 15, 15] quantization scale improves details in images, at an imperceptable cost to color accuracy.


### Bits per pixel and color scale

![rgb](https://github.com/theAdamColton/spiht-py/assets/72479734/582b7599-12a1-424d-96dd-0e35848c34a9)
![ipt](https://github.com/theAdamColton/spiht-py/assets/72479734/5178e3ce-b618-4198-9c00-e36643e14c99)

Identical BPP between the two columns. The left column uses the RGB color space. The right column uses the IPT color space.
BPP are as follows, from top to bottom row: 0.075, 0.1, 0.5, 1.0

You can see that the image compressed using the IPT color space has better fine details at all compression levels.

# Tests

I set up some simple python and rust tests. You can run the rust test using `cargo test`. The python tests can be run with `python -m unittest`

# Potential performance hacks

I did not play around with any performance hacks that might speed up the encoding. For example, the test for significance could be run in parallel for all pixels at the start of an encoding iteration.

Another hack would be to use a 'bit-significance-wise' data structure to store the DWT coefficients. A naive approach just uses a raster ordered storage of i32s. A better way would be some sort of custom datastructure that takes advantage of the particular memory access patterns of SPIHT. Bits of the coefficient array are traversed strictly in order of significance and then in DWT hierarchical ordering.
