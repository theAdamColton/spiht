use bitvec::field::BitField;
use bitvec::vec::BitVec;
use ndarray::Dim;
use numpy::{PyArray, PyReadonlyArray3, ToPyArray};
use pyo3::{
    pymodule,
    pyfunction,
    types::{PyModule, PyBytes},
    PyResult, Python, PyObject, wrap_pyfunction,
};

use encoder_decoder::{decode, decode_with_metadata, encode, Slices};
mod encoder_decoder;

fn bytes_to_bits(bytes: Vec<u8>) -> BitVec {
    let mut data = BitVec::repeat(false, bytes.len() * 8);
    for (slot, byte) in data.chunks_mut(8).zip(bytes.into_iter()) {
      slot.store_le(byte);
    }
    data
}

/// Encode DWT coefficients into bytes
#[pyfunction]
#[pyo3(name="encode")]
fn encode_spiht(py: Python, x: PyReadonlyArray3<i32>, ll_h: usize, ll_w: usize, max_bits: usize) -> (PyObject, u8) {
    let x = x.as_array();
    let (data, max_n) = encode(x, ll_h, ll_w, max_bits);
    let vec: Vec<u8> = data.chunks(8).map(BitField::load_le::<u8>).collect();
    let bytes = PyBytes::new(py, &vec);
    (bytes.into(), max_n)
}

/// Decode DWT coefficients from bytes
#[pyfunction]
#[pyo3(name="decode")]
fn decode_spiht<'py>(py: Python<'py>, data_u8: Vec<u8>, n: u8, c: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> &'py PyArray<i32, Dim<[usize; 3]>> {
    let data = bytes_to_bits(data_u8);

    let rec_arr = decode(data, n, c, h, w, ll_h, ll_w);
    rec_arr.to_pyarray(py)
}

/// Decode DWT coefficients from bytes
/// Also return intermediate metadata
/// of the spiht algorithm
#[pyfunction]
#[pyo3(name="decode_with_metadata")]
fn decode_spiht_with_metadata<'py>(py:Python<'py>, data_u8: Vec<u8>, n: u8, c: usize, h: usize, w:usize, ll_h: usize, ll_w: usize, top_slice: Vec<(usize, usize)>, other_slices: Vec<Vec<Vec<(usize, usize)>>>) -> (&'py PyArray<i32, Dim<[usize; 3]>>, &'py PyArray<i32, Dim<[usize; 2]>>){
    let data = bytes_to_bits(data_u8);

    let slices = Slices::from_vec(top_slice, other_slices);

    let (rec_arr,metadata_arr) = decode_with_metadata(data, n, c, h, w, ll_h, ll_w, slices);
    (rec_arr.to_pyarray(py), metadata_arr.to_pyarray(py))
}

#[pymodule]
fn spiht<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_spiht, m)?)?;
    m.add_function(wrap_pyfunction!(decode_spiht, m)?)?;
    m.add_function(wrap_pyfunction!(decode_spiht_with_metadata, m)?)?;

    Ok(())
}

