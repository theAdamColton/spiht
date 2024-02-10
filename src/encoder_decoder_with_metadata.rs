use bitvec::vec::BitVec;

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

///// h: size of rec array
///// w: size of rec array
///// i: global position of coeff in rec array
///// j: global position of coeff in rec array
///// slices: rec array slices
///// level: level setting for initial DWT
//fn get_relative_metadata_of_coeff(
//    h: usize,
//    w: usize,
//    i: usize,
//    j: usize,
//    slices: Slices,
//    level: u8,
//    ) -> (u8, u8, usize, usize) {
//
//    let mut depth = level;
//
//    if slices.top_slice.contains(i, j) {
//        return (depth, Filter::LL as u8, i, j);
//    }
//
//    // searches for the slice that i,j is in
//    // TODO this can be sped up by binary search
//    for slice_i in 0..slices.other_slices.len() {
//        let slice_curr = slices.other_slices[slice_i];
//
//        if slice_i + 1 < slices.other_slices.len() {
//            let slice_next = slices.other_slices[slice_i + 1];
//            // needs to check the boundary case,
//            // where i,j is in a padded area,
//            // and is in between s_d and s_d_next
//
//            // test for 'da'
//            let da_slice = Slice {
//                start_i: slice_curr.da.start_i,
//                end_i: slice_next.da.start_i,
//                // 'da' always starts from col 0
//                start_j: 0,
//                end_j: slice_curr.dd.start_j,
//            };
//            if da_slice.contains(i, j) {
//                return (depth, Filter::DA as u8, i - da_slice.start_i, j)
//            }
//
//            // test for 'ad'
//            // 'ad' starts at row 0
//            let ad_slice = Slice {
//                start_i: 0,
//                end_i: slice_curr.dd.start_i,
//                start_j: slice_curr.ad.start_j,
//                end_j: slice_next.ad.start_j,
//            };
//            if ad_slice.contains(i, j) {
//                return (depth, Filter::AD as u8, i, j-ad_slice.start_j)
//            }
//
//            // test for 'dd'
//            let dd_slice = Slice {
//                start_i: slice_curr.dd.start_i,
//                end_i: slice_next.da.start_i,
//                start_j: slice_curr.dd.start_j,
//                end_j: slice_next.ad.start_j,
//            };
//            if dd_slice.contains(i, j) {
//                return (depth, Filter::DD as u8, i - dd_slice.start_i, j-dd_slice.start_j);
//            }
//
//        } else {
//            // test for 'da'
//            let da_slice = Slice {
//                start_i: slice_curr.da.start_i,
//                end_i: h,
//                start_j: 0,
//                end_j: slice_curr.dd.start_j
//            };
//            if da_slice.contains(i, j) {
//                return (depth, Filter::DA as u8, i - da_slice.start_i, j);
//            }
//
//            // test for 'ad'
//            let ad_slice = Slice {
//                start_i: 0,
//                end_i: slice_curr.dd.start_i,
//                start_j: slice_curr.ad.start_j,
//                end_j: w
//            };
//            if ad_slice.contains(i, j) {
//                return (depth, Filter::AD as u8, i, j - ad_slice.start_j);
//            }
//
//            // test for 'dd'
//            let dd_slice = Slice {
//                start_i: slice_curr.dd.start_i, 
//                end_i: h,
//                start_j: slice_curr.dd.start_j,
//                end_j: w
//            };
//            if dd_slice.contains(i, j) {
//                return (depth, Filter::DD as u8, i - dd_slice.start_i, j - dd_slice.start_j);
//            }
//        }
//
//        depth -= 1;
//
//    }
//
//    panic!();
//}

pub fn decode_with_metadata(
    data:BitVec,
    mut n:u8,
    c: usize,
    h:usize,
    w:usize,
    slices: Slices,
    ) {
}
