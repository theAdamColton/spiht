use std::ops::Add;

use bitvec::field::BitField;
use bitvec::vec::BitVec;
use ndarray::Dim;
use numpy::{PyArray3, PyArray, PyReadonlyArray3, ToPyArray};
use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::types::PyList;
use pyo3::{
    exceptions::PyIndexError,
    pymodule,
    types::{PyDict, PyModule},
    FromPyObject, PyAny, PyObject, PyResult, Python,
};

use spiht::{encode, decode};
mod spiht;


#[pymodule]
fn spiht_rs<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    // example using generic PyObject
    fn encode_spiht(x: PyReadonlyArray3<i32>, ll_h: usize, ll_w: usize, max_bits: usize) -> (Vec<u8>, u8) {
        let x = x.as_array();
        let (data, max_n) = encode(x, ll_h, ll_w, max_bits);
        let vec = data.chunks(8).map(BitField::load_le::<u8>).collect();
        (vec, max_n)
    }

    #[pyfn(m)]
    // example using generic PyObject
    fn decode_spiht<'py>(py: Python<'py>, data_u8: Vec<u8>, n: u8, c: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> &'py PyArray<i32, Dim<[usize; 3]>> {
        let mut data = BitVec::repeat(false, data_u8.len() * 8);
        for (slot, byte) in data.chunks_mut(8).zip(data_u8.into_iter()) {
          slot.store_le(byte);
        }

        let rec_arr = decode(data, n, c, h, w, ll_h, ll_w);
        rec_arr.to_pyarray(py)
    }

    Ok(())
}

