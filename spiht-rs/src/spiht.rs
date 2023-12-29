use std::collections::VecDeque;

use ndarray::{ArrayView3, Array3};
use ndarray_stats::QuantileExt;
use bitvec::vec::BitVec;

fn set_bit(x: i32, n: u8, bit: bool) -> i32 {
    let sign = x >= 0;
    if bit {
        if sign {
            x | 1 << n
        } else {
            -((-x) | 1 << n)
        }
    } else {
        if sign {
            x & !(1<<n)
        } else {
            -((-x) & !(1<<n))
        }
    }
}

fn is_bit_set(x: i32, n: u8) -> bool
{
    (x.abs() & (1 << n)) != 0
}


fn is_element_sig(x: i32, n: u8)  -> bool
{
    debug_assert!(n < 32);
    x.abs() >= (1i32 <<n)
}

fn get_offspring(i: usize, j: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Vec<(usize, usize)>{
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

fn is_set_sig(arr: ArrayView3<i32>,k: usize,i: usize,j:usize,n:u8, ll_h: usize, ll_w: usize) -> bool {
    let shape = arr.shape();

    let h = shape[shape.len() -2];
    let w = shape[shape.len() - 1];

    if is_element_sig(arr[(k,i,j)], n) {
        return true
    }

    for (l,m) in get_offspring(i,j,h,w, ll_h, ll_w) {
        if is_set_sig(arr,k,l,m,n,ll_h, ll_w) {
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

        //println!("encoding, n = {}, {}kb", n, (data.len() / 8) / 1024);

        while let Some((k,i,j)) = queue.pop_front() {
            let x = arr[(k,i,j)];

            let is_already_sig = sig_sets[(k,i,j)] == 1;

            let is_currently_sig: bool;
            if is_already_sig {
                is_currently_sig = true;
            } else {
                is_currently_sig = is_set_sig(arr,k,i,j,n,ll_h, ll_w);
            }

            if !is_already_sig {
                data.push(is_currently_sig);

                if data.len() == max_bits {
                    return (data, max_n)
                }
            }

            let is_newly_sig = !is_already_sig && is_currently_sig;

            //println!("n{} x{} {} {} {} is_already_sig {} is_currently_sig {} is_newly_sig {} is_bit_set {}",n, x, k,i,j,is_already_sig, is_currently_sig, is_newly_sig, is_bit_set(x, n));


            if is_currently_sig {
                let offspring = get_offspring(i,j,h,w,ll_h,ll_w);

                for (l,m) in offspring {
                    queue.push_back((k,l,m));
                }
            }

            if is_newly_sig {

                let element_sig = is_element_sig(x, n);

                data.push(element_sig);
                if data.len() == max_bits {
                    return (data, max_n)
                }

                if element_sig {
                    sig_sets[(k,i,j)] = 1;

                    let sign = x >= 0;

                    data.push(sign);
                    if data.len() == max_bits {
                        return (data, max_n)
                    }
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

pub fn decode(data: BitVec, mut n: u8, c:usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Array3<i32> {
    let mut rec_arr = Array3::<i32>::zeros((c,h,w));

    let mut sig_pixels = Array3::<u8>::zeros((c, h, w));

    let mut cur_i = 0;

    let mut pop_front = || {
        let ret: Option<bool>;
        if cur_i >= data.len() {
            ret = None
        } else {
            ret = Some(data[cur_i]);
        }
        cur_i += 1;
        return ret
    };

    loop {
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

        for i in 0..ll_h {
            for j in 0..ll_w {
                for k in 0..c {
                    queue.push_back((k,i,j));
                }
            }
        }

        //println!("encoding, n = {}, {}kb", n, (data.len() / 8) / 1024);

        while let Some((k,i,j)) = queue.pop_front() {
            let is_already_sig:bool = sig_pixels[(k,i,j)] == 1;

            let is_currently_sig: bool;
            if is_already_sig {
                is_currently_sig = true;
            } else {
                if let Some(x) = pop_front() {
                    is_currently_sig = x; 
                } else {
                    return rec_arr
                }
            }

            let is_newly_sig = !is_already_sig && is_currently_sig;

            //println!("n{} {} {} {} is_already_sig {} is_currently_sig {} is_newly_sig {}",n, k,i,j,is_already_sig, is_currently_sig, is_newly_sig);
            //

            if is_currently_sig {
                let offspring = get_offspring(i,j,h,w,ll_h,ll_w);

                for (l,m) in offspring {
                    queue.push_back((k,l,m));
                }
            }

            if is_newly_sig {
                
                let element_sig: bool;
                if let Some(x) = pop_front() {
                    element_sig = x;
                } else {
                    return rec_arr;
                }

                if element_sig {
                    sig_pixels[(k,i,j)] = 1;

                    let sign: i32;
                    if let Some(x) = pop_front() {
                        // 1 or -1
                        sign = x as i32 * 2 - 1;
                        //println!("{}", sign);
                    } else {
                        return rec_arr;
                    }

                    let base_sig: i32;
                    if n==0 {
                        base_sig = 1<<n;
                    } else {
                        // should be eq to 1.5 * 2 ^ n
                        base_sig = (1 << (n-1)) + (1<<n);
                    }
                    //println!("initializing {} {} {} to {}", k,i,j, sign*base_sig);
                    rec_arr[(k,i,j)] = sign * base_sig;
                }
            }

            if is_already_sig {
                let bit:bool;
                if let Some(x) = pop_front() {
                    bit = x;
                } else {
                    return rec_arr
                }

                //println!("b4 {}", rec_arr[(k,i,j)]);

                rec_arr[(k,i,j)] = set_bit(rec_arr[(k,i,j)], n, bit);

                //println!("after set bit {} {} {}",n,bit, rec_arr[(k,i,j)]);
            }
        } 


        if n == 0 {
            return rec_arr
        }
        n -= 1;
    }
}



#[cfg(test)]
mod tests {
    use ndarray_rand::{RandomExt, rand::{rngs::SmallRng, SeedableRng, Rng}};
    use ndarray_rand::rand_distr::Normal;

    use super::*;
    #[test]
    fn test_is_bit_set() {
        assert!(is_bit_set(32, 5));
        assert_eq!(is_bit_set(32, 4), false);
        assert_eq!(is_bit_set(32, 3), false);
        assert_eq!(is_bit_set(32, 2), false);
        assert_eq!(is_bit_set(32, 1), false);
        assert_eq!(is_bit_set(32, 0), false);
        println!("{:08b}", -69i32 & (1<<6 as i32));
        assert!(is_bit_set(-69, 6));
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
        //println!("{}",data);
    }

    #[test]
    fn simple_test_encode_decode() {
        let ll_h = 2;
        let ll_w = 2;
        let h = 16;
        let w = 16;
        let c = 1;
        let arr: Array3<i32> = Array3::ones((c,h,w)) * 32;
        let (data, max_n) = encode(arr.view(), ll_h, ll_w, 10000);
        let rec_data = decode(data, max_n, c, h, w, ll_h, ll_w);
        assert_eq!(arr, rec_data);
    }

    #[test]
    fn simple_test_encode_decode_w_negative() {
        let ll_h = 2;
        let ll_w = 2;
        let h = 16;
        let w = 16;
        let c = 1;
        let mut arr: Array3<i32> = Array3::ones((c,h,w)) * 32;
        for k in 0..c {
            for i in 0..h {
                for j in 0..w {
                    let sign = (i%2 != 0) as i32 * 2 - 1;
                    arr[(k,i,j)] *= sign;
                }
            }
        }
        let (data, max_n) = encode(arr.view(), ll_h, ll_w, 10000);
        let rec_data = decode(data, max_n, c, h, w, ll_h, ll_w);
        assert_eq!(arr, rec_data);
    }

    #[test]
    fn test_encode_decode_many_random() {
        let rng = &mut SmallRng::seed_from_u64(42);

        let ll_h = 2;
        let ll_w = 2;
        let h = 8;
        let w = 8;
        let c = 1;
        for _ in 0..10 {
            let arrf = Array3::random_using((c,h,w), Normal::new(0.,16.).unwrap(), rng);
            let arr = arrf.mapv(|x| x as i32);
            let (data, max_n) = encode(arr.view(), ll_h, ll_w, 10000000);
            let rec_data = decode(data, max_n, c, h, w, ll_h, ll_w);
            assert_eq!(arr, rec_data);
        }
    }

    #[test]
    fn test_base_sig() {
        assert_eq!((1<<5) + (1<<6), 96);
        assert_eq!((1<<4) + (1<<5), 48)
    }

    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(-96, 5, false), -64);
        assert_eq!(set_bit(-96, 5, true), -96);
        assert_eq!(set_bit(-64, 5, true), -96);
        assert_eq!(set_bit(96, 5, true), 96);
        assert_eq!(set_bit(96, 5, false), 64);
    }

    #[test]
    fn test_is_element_sig() {
        // 2^6 == 64
        assert_eq!(is_element_sig(-21, 6), false);
        assert_eq!(is_element_sig(-64, 6), true);
        assert_eq!(is_element_sig(64, 6), true);
        assert_eq!(is_element_sig(55, 6), false);
    }

    #[test]
    fn test_set_bit_doesnt_change_sign() {
        let mut rng = SmallRng::seed_from_u64(420);
        for _ in 0..420 {
            let x:i32 = rng.gen();
            let sign = x >= 0;
            let n:u8 = rng.gen_range(0..16);
            let bit: bool = rng.gen_bool(0.5);
            let y = set_bit(x, n, bit);
            let sign_y = y >= 0;
            assert_eq!(sign, sign_y);
            assert_eq!(is_bit_set(y, n), bit);
        }
    }
}
