use std::collections::VecDeque;

use ndarray::{ArrayView3, Array3};
use ndarray_stats::QuantileExt;
use bitvec::vec::BitVec;

fn has_descendents_past_offspring(i:usize,j:usize,h:usize,w:usize) -> bool {
    if 2*i + 1 >= h || 2*j + 1 >= w {
        return false
    } 
    return true
}

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

fn get_offspring(i: usize, j: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Option<[(usize,usize);4]> {
    if i < ll_h && j < ll_w {
        if i%2 == 0 && j%2 == 0 {
            return None;
        }
        // index relative to the top left chunk corner
        // can be (0,0), (0,2), (2,0), (2,2)
        let sub_child_i = i / 2 * 2;
        let sub_child_j = j/2 * 2;

        // can be (0,1), (1,0) or (1,1)
        let chunk_i = i % 2;
        let chunk_j = j%2;

        return Some([
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j + 1),
                (chunk_i * ll_h + sub_child_i+1, chunk_j * ll_w + sub_child_j),
                (chunk_i * ll_h + sub_child_i + 1, chunk_j * ll_w + sub_child_j + 1),
            ])
    }

    if 2*i+1 >= h || 2*j+1 >= w {
        return None;
    }

    return Some([
            (2*i,2*j), 
            (2*i, 2*j+1),
            (2*i+1, 2*j),
            (2*i+1, 2*j+1),
        ])
}


//fn get_offspring(i: usize, j: usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Vec<(usize, usize)>{
//    if i < ll_h && j < ll_w {
//        if i%2 == 0 && j%2 == 0 {
//            return vec![];
//        }
//        // index relative to the top left chunk corner
//        // can be (0,0), (0,2), (2,0), (2,2)
//        let sub_child_i = i / 2 * 2;
//        let sub_child_j = j/2 * 2;
//
//        // can be (0,1), (1,0) or (1,1)
//        let chunk_i = i % 2;
//        let chunk_j = j%2;
//
//        return vec![
//                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j),
//                (chunk_i * ll_h + sub_child_i, chunk_j * ll_w + sub_child_j + 1),
//                (chunk_i * ll_h + sub_child_i+1, chunk_j * ll_w + sub_child_j),
//                (chunk_i * ll_h + sub_child_i + 1, chunk_j * ll_w + sub_child_j + 1),
//                ]
//    }
//
//    if 2*i+1 >= h || 2*j+1 >= w {
//        return vec![]
//    }
//
//    return vec![
//            (2*i,2*j), 
//            (2*i, 2*j+1),
//            (2*i+1, 2*j),
//            (2*i+1, 2*j+1),
//        ]
//}
//
fn is_set_sig(arr: ArrayView3<i32>,k: usize,i: usize,j:usize,n:u8, ll_h: usize, ll_w: usize) -> bool {
    let shape = arr.shape();

    let h = shape[shape.len() -2];
    let w = shape[shape.len() - 1];

    if is_element_sig(arr[(k,i,j)], n) {
        return true
    }

    let offspring = get_offspring(i,j,h,w, ll_h, ll_w);

    if let Some(offspring) = offspring {
        for (l,m) in offspring {
            if is_set_sig(arr,k,l,m,n,ll_h, ll_w) {
                return true
            }
        }
    }

    false
}

fn is_l_sig(arr: ArrayView3<i32>,k: usize,i: usize,j:usize,n:u8, ll_h: usize, ll_w: usize) -> bool {
    let shape = arr.shape();

    let h = shape[shape.len() -2];
    let w = shape[shape.len() - 1];

    let offspring = get_offspring(i, j, h, w, ll_h, ll_w);
    if let Some(offspring) = offspring {
        for (l,m) in offspring {
            let secondary_offspring = get_offspring(l, m, h, w, ll_h, ll_w);
            if let Some(secondary_offspring) = secondary_offspring {
                for (ll,mm) in secondary_offspring {
                    if is_set_sig(arr, k, ll, mm, n, ll_h, ll_w) {
                        return true
                    }
                }
            }
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

    let mut lsp: VecDeque<(usize,usize,usize)> = VecDeque::new();
    let mut lip: VecDeque<(usize,usize,usize)> = VecDeque::new();
    for i in 0..ll_h {
        for j in 0..ll_w {
            for k in 0..c {
                lip.push_back((k,i,j));
            }
        }
    }

    // true=type A, false = type B
    let mut lis: VecDeque<(bool,usize,usize,usize)> = VecDeque::new();
    for i in 0..ll_h {
        for j in 0..ll_w {
            if i%2 == 0 && j%2 == 0 {
                continue;
            }
            for k in 0..c {
                lis.push_back((true, k,i,j));
            }
        }
    }

    
    loop {
        let lsp_len = lsp.len();

        let mut lip_retain: VecDeque<(usize,usize,usize)> = VecDeque::new();
        for (k,i,j) in lip {
            let x = arr[(k,i,j)];
            let is_sig = is_element_sig(x, n);
            data.push(is_sig);
            if data.len() == max_bits {
                return (data, max_n)
            }

            if is_sig {
                lsp.push_back((k,i,j));

                let sign = x >=0;
                data.push(sign);
                if data.len() == max_bits {
                    return (data, max_n)
                }
            } else {
                lip_retain.push_back((k,i,j));
            }
        }
        lip = lip_retain;

        let mut lis_retain: VecDeque<(bool,usize,usize,usize)> = VecDeque::new();
        while let Some((t,k,i,j)) = lis.pop_front() {
            if t {
                // type A
                let mut desc_sig = false;
                let offspring = get_offspring(i,j,h,w,ll_h,ll_w);
                if let Some(offspring) = offspring {
                    for (l,m) in offspring {
                        if is_set_sig(arr, k, l, m, n, ll_h, ll_w) {
                            desc_sig = true;
                            break;
                        }
                    }
                }

                data.push(desc_sig);
                if data.len() == max_bits {
                    return (data, max_n)
                }

                if desc_sig {
                    for (l,m) in offspring.unwrap() {
                        let sig = is_element_sig(arr[(k,l,m)], n);

                        data.push(sig);
                        if data.len() == max_bits {
                            return (data, max_n)
                        }

                        if sig {
                            lsp.push_back((k,l,m));

                            let sign = arr[(k,l,m)] >= 0;
                            data.push(sign);
                            if data.len() == max_bits {
                                return (data, max_n)
                            }
                        } else {
                            lip.push_back((k,l,m));
                        }
                    }


                    let l_exists = has_descendents_past_offspring(i,j,h,w);
                    if l_exists {
                        lis.push_back((false, k,i,j));
                    }
                } else {
                    lis_retain.push_back((t,k,i,j));
                }


            } else {
                // type B
                let l_sig = is_l_sig(arr, k, i, j, n, ll_h, ll_w);
                data.push(l_sig);
                if data.len() == max_bits {
                    return (data, max_n)
                }

                if l_sig {
                    let offspring = get_offspring(i, j, h, w, ll_h, ll_w);
                    if let Some(offspring) = offspring {
                        for (l,m) in offspring {
                            lis.push_back((true,k,l,m));
                        }
                    }
                } else {
                    lis_retain.push_back((t,k,i,j));
                }
            }
        }
        lis = lis_retain;

        // refinement
        for lsp_i in 0..lsp_len {
            let (k,i,j) = lsp[lsp_i];
            let bit = is_bit_set(arr[(k,i,j)], n);

            data.push(bit);
            if data.len() == max_bits {
                return (data, max_n)
            }
        }


        if n==0 {
            break;
        }

        n-= 1;
    }

    return (data, max_n);
}



pub fn decode(data: BitVec, mut n: u8, c:usize, h: usize, w: usize, ll_h: usize, ll_w: usize) -> Array3<i32> {
    let mut rec_arr = Array3::<i32>::zeros((c,h,w));

    assert!(ll_h > 1);
    assert!(ll_w > 1);

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

    let mut lsp: VecDeque<(usize,usize,usize)> = VecDeque::new();
    let mut lip: VecDeque<(usize,usize,usize)> = VecDeque::new();
    for i in 0..ll_h {
        for j in 0..ll_w {
            for k in 0..c {
                lip.push_back((k,i,j));
            }
        }
    }

    // true=type A, false = type B
    let mut lis: VecDeque<(bool,usize,usize,usize)> = VecDeque::new();
    for i in 0..ll_h {
        for j in 0..ll_w {
            if i%2 == 0 && j%2 == 0 {
                continue;
            }
            for k in 0..c {
                lis.push_back((true, k,i,j));
            }
        }
    }


    
    loop {
        let lsp_len = lsp.len();

        let mut lip_retain: VecDeque<(usize,usize,usize)> = VecDeque::new();
        for (k,i,j) in lip {
            let is_sig:bool;
            if let Some(x) = pop_front() {
                is_sig=x;
            } else {
                return rec_arr
            }

            if is_sig {
                lsp.push_back((k,i,j));

                let sign:i32;
                if let Some(x) = pop_front() {
                    // -1 or 1
                    sign = x as i32 * 2 - 1;
                } else {
                    return rec_arr
                }

                let base_sig: i32;
                if n==0 {
                    base_sig = 1<<n;
                } else {
                    // should be eq to 1.5 * 2 ^ n
                    base_sig = (1 << (n-1)) + (1<<n);
                }

                rec_arr[(k,i,j)] = base_sig * sign;
            } else {
                lip_retain.push_back((k,i,j));
            }
        }
        lip = lip_retain;

        let mut lis_retain: VecDeque<(bool,usize,usize,usize)> = VecDeque::new();
        while let Some((t,k,i,j)) = lis.pop_front() {
            if t {
                // type A
                let desc_sig:bool;
                if let Some(x) = pop_front() {
                    desc_sig = x;
                } else {
                    return rec_arr
                }

                if desc_sig {
                    let offspring = get_offspring(i, j, h, w, ll_h, ll_w);
                    if let Some(offspring) = offspring {
                        for (l,m) in offspring {
                            let sig: bool;
                            if let Some(x) = pop_front() {
                                sig = x;
                            } else {
                                return rec_arr
                            }

                            if sig {
                                lsp.push_back((k,l,m));

                                let sign: i32;
                                if let Some(x) = pop_front() {
                                    // -1 or 1
                                    sign = x as i32 * 2 -1;
                                } else {
                                    return rec_arr
                                }

                                let base_sig: i32;
                                if n==0 {
                                    base_sig = 1<<n;
                                } else {
                                    // should be eq to 1.5 * 2 ^ n
                                    base_sig = (1 << (n-1)) + (1<<n);
                                }

                                rec_arr[(k,l,m)] = sign * base_sig;
                            } else {
                                lip.push_back((k,l,m));
                            }
                        }
                    }

                    let l_exists = has_descendents_past_offspring(i,j,h,w);
                    if l_exists {
                        lis.push_back((false, k,i,j));
                    }
                } else {
                    lis_retain.push_back((t,k,i,j));
                }


            } else {
                // type B
                let l_sig: bool;
                if let Some(x) = pop_front() {
                    l_sig = x;
                } else {
                    return rec_arr
                }

                if l_sig {
                    let offspring = get_offspring(i, j, h, w, ll_h, ll_w);
                    if let Some(offspring) = offspring {
                        for (l,m) in offspring {
                            lis.push_back((true,k,l,m));
                        }
                    }
                } else {
                    lis_retain.push_back((t,k,i,j));
                }
            }
        }
        lis = lis_retain;

        // refinement
        for lsp_i in 0..lsp_len {
            let (k,i,j) = lsp[lsp_i];
            let bit:bool;
            if let Some(x) = pop_front() {
                bit = x;
            } else {
                return rec_arr;
            }

            rec_arr[(k,i,j)] = set_bit(rec_arr[(k,i,j)], n,bit);
        }

        if n==0 {
            break;
        }

        n-= 1;
    }

    rec_arr
}




pub struct Slice{
    start_i: usize,
    end_i: usize,
    start_j: usize,
    end_j: usize,
}

impl Slice {
    pub fn contains(&self, i: usize, j: usize) -> bool {
        i >= self.start_i && i < self.end_i && j >= self.start_j && j < self.end_j
    }
}

pub struct OtherSlice {
    da: Slice,
    ad: Slice,
    dd: Slice
}

pub struct Slices{
    top_slice: Slice,
    other_slices:Vec<OtherSlice>
}

enum Filter {
    LL = 0,
    DA = 1,
    AD = 2,
    DD = 3
}

pub fn decode_with_metadata(
    data:BitVec,
    mut n:u8,
    c: usize,
    h:usize,
    w:usize,
    slices: Slices,
    ) {
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
    fn test_encode_decode_many_random_large() {
        let rng = &mut SmallRng::seed_from_u64(42);

        let ll_h = 2;
        let ll_w = 2;
        let h = 32;
        let w = 32;
        let c = 4;
        for _ in 0..20 {
            let arrf = Array3::random_using((c,h,w), Normal::new(0.,16.).unwrap(), rng);
            let arr = arrf.mapv(|x| x as i32);
            let (data, max_n) = encode(arr.view(), ll_h, ll_w, 10000000);
            let rec_data = decode(data, max_n, c, h, w, ll_h, ll_w);
            assert_eq!(arr, rec_data);
        }
    }

    #[test]
    fn test_encode_decode_many_random() {
        let rng = &mut SmallRng::seed_from_u64(42);

        let ll_h = 2;
        let ll_w = 2;
        let h = 8;
        let w = 8;
        let c = 1;
        for _ in 0..20 {
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
