use std::collections::VecDeque;

use ndarray::{ArrayView3, Array3};
use ndarray_stats::QuantileExt;
use bitvec::{bitvec, BitArr, vec::BitVec};

pub fn is_bit_set(x: i32, n: u8) -> bool
{
    x & (1 << n) != 0
}


pub fn is_element_sig(x: i32, n: u8)  -> bool
{
    x.abs() >= (1i32 <<n)
}

pub fn get_offspring(i: usize, j: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Vec<(usize, usize)>{
    if i < ll_h && j < ll_w {
        if i%2 == 0 && j%2 == 0 {
            return vec![];
        }
        // index relative to the top left chunk corner
        // can be (0,0), (0,2), (2,0), (2,2)
        let sub_child_i = i / 2 * 2;
        let sub_child_j = j/2 * 2;

        // can be (0,1), (1,0) or (1,1)
        let chunk_i = i % 2;
        let chunk_j = j%2;

        return vec![
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j + 1),
                (chunk_i * ll_h + sub_child_i+1, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i + 1, chunk_j * ll_w + sub_child_j + 1),
                ]
    }

    if 2*i+1 >= h || 2*j+1 >= w {
        return vec![]
    }

    return vec![
            (2*i,2*j), 
            (2*i, 2*j+1),
            (2*i+1, 2*j),
            (2*i+1, 2*j+1),
        ]
}

pub fn is_set_sig(arr: ArrayView3<i32>,k: usize,i: usize,j:usize,n:u8, ll_h: usize, ll_w: usize) -> bool {
    let shape = arr.shape();

    let h = shape[shape.len() -2];
    let w = shape[shape.len() - 1];

    if is_element_sig(arr[(k,i,j)], n) {
        return true
    }

    for (i,j) in get_offspring(i,j,h,w, ll_h, ll_w) {
        if is_set_sig(arr,k,i,j,n,ll_h, ll_w) {
            return true
        }
    }

    false
}

pub fn encode(arr: ArrayView3<i32>, ll_h: usize, ll_w: usize, max_bits: usize) -> (BitVec, u8) {
    let c = arr.shape()[0];
    let h = arr.shape()[1];
    let w = arr.shape()[2];

    assert!(ll_h > 1);
    assert!(ll_w > 1);


    let mut data = BitVec::new();
    let max = *arr.mapv(i32::abs).max().unwrap();
    let mut n = (max as f32).log2() as u8;
    let max_n = n;

    let mut sig_sets = Array3::<u8>::zeros((c, h, w));

    loop {
        
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

        for i in 0..ll_h {
            for j in 0..ll_w {
                for k in 0..c {
                    queue.push_back((k,i,j));
                }
            }
        }

        println!("encoding, n = {}, {}kb", n, (data.len() / 8) / 1024);

        while let Some((k,i,j)) = queue.pop_front() {
            let x = arr[(k,i,j)];

            let is_already_sig:bool = sig_sets[(k,i,j)] == 1;

            let is_currently_sig: bool;
            if is_already_sig {
                is_currently_sig = true;
            } else {
                is_currently_sig = is_set_sig(arr,k,i,j,n,ll_h, ll_w);
            }

            let is_newly_sig = !is_already_sig && is_currently_sig;

            println!("n{} {} {} {} is_already_sig {} is_currently_sig {} is_newly_sig {}",n, k,i,j,is_already_sig, is_currently_sig, is_newly_sig);

            if !is_already_sig {
                data.push(is_currently_sig);

                if data.len() == max_bits {
                    return (data, max_n)
                }
            }

            if is_currently_sig {
                let offspring = get_offspring(i,j,h,w,ll_h,ll_w);

                for (l,m) in offspring {
                    queue.push_back((k,l,m));
                }
            }

            if is_newly_sig {
                sig_sets[(k,i,j)] = 1;

                let element_sig = is_element_sig(x, n);

                data.push(element_sig);
                if data.len() == max_bits {
                    return (data, max_n)
                }

                let sign = x >= 0;

                data.push(sign);
                if data.len() == max_bits {
                    return (data, max_n)
                }
            }

            if is_already_sig {
                let bit = is_bit_set(x, n);

                data.push(bit);
                if data.len() == max_bits {
                    return (data, max_n)
                }
            }
        } 


        if n == 0 {
            return (data, max_n)
        }
        n -= 1;
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_bit_set() {
        assert!(is_bit_set(32, 5));
        assert_eq!(is_bit_set(32, 4), false);
        assert_eq!(is_bit_set(32, 3), false);
        assert_eq!(is_bit_set(32, 2), false);
        assert_eq!(is_bit_set(32, 1), false);
        assert_eq!(is_bit_set(32, 0), false);
        assert_eq!(is_bit_set(3590854, 8), false);
    }

    #[test]
    fn simple_test_encode() {
        let ll_h = 2;
        let ll_w = 2;
        let h = 16;
        let w = 16;
        let c = 1;
        let arr: Array3<i32> = Array3::ones((c,h,w)) * 32;
        let (data, max_n) = encode(arr.view(), ll_h, ll_w, 10000);
        assert_eq!(max_n, 5);
        println!("{}",data);
    }

}
