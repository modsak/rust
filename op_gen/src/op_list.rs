use std::slice;
use tensorflow_sys as tf;

use tensorflow_protos::op_def::OpList;
use protobuf;

struct TFByteBuffer {
    inner: *mut tf::TF_Buffer,
}

impl TFByteBuffer {
    unsafe fn from_c(buf: *mut tf::TF_Buffer) -> Self {
        Self {
            inner: buf,
        }
    }

    #[inline]
    fn data(&self) -> *const u8 {
        unsafe { (*self.inner).data as *const u8 }
    }
}

impl AsRef<[u8]> for TFByteBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data(), (*self.inner).length) }
    }
}

impl Drop for TFByteBuffer {
    fn drop(&mut self) {
        unsafe {
            tf::TF_DeleteBuffer(self.inner);
        }
    }
}

pub fn load_oplist() -> OpList {
    let buf: TFByteBuffer;
    unsafe {
        buf = TFByteBuffer::from_c(tf::TF_GetAllOpList());
    }
    protobuf::parse_from_bytes(buf.as_ref()).unwrap()
}