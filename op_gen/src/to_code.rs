use tensorflow_protos::tensor_shape::TensorShapeProto;
use tensorflow_protos::tensor::{TensorProto, VariantTensorDataProto};
use tensorflow_protos::attr_value::AttrValue_ListValue;
use tensorflow_protos::resource_handle::ResourceHandleProto;
use std::f64;
use std::f32;

pub(crate) trait ToCode {
    fn to_code(&self) -> Result<String, String>;
}

impl ToCode for TensorShapeProto {
    fn to_code(&self) -> Result<String, String> {
        if self.get_unknown_rank() {
            return Ok("None.into()".to_string());
        }

        if self.get_dim().len() == 0 {
            return Ok("Some(vec![]).into()".to_string());
        }

        let mut rtn = "Some(vec![".to_string();
        for dim in self.get_dim() {
            if dim.get_size() == -1 {
                rtn.push_str("None,");
            } else {
                rtn.push_str(&format!("Some({}),", dim.get_size()));
            }
        }
        rtn.pop();
        rtn.push_str("]).into()");
        Ok(rtn)
    }
}

impl<T: ToCode> ToCode for Vec<T> {
    fn to_code(&self) -> Result<String, String> {
        let mut rtn = "vec![".to_string();
        for elem in self {
            rtn.push_str(&format!("{}, ", &elem.to_code()?));
        }

        if self.len() != 0 {
            rtn.pop();
        }

        rtn.push_str("]");
        Ok(rtn)
    }
}

impl<T: ToCode> ToCode for [T] {
    fn to_code(&self) -> Result<String, String> {
        let mut rtn = "&[".to_string();
        for elem in self {
            rtn.push_str(&format!("{}, ", &elem.to_code()?));
        }

        if self.len() != 0 {
            rtn.pop();
        }

        rtn.push_str("]");
        Ok(rtn)
    }
}

impl ToCode for u8 {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}_u8", self))
    }
}

impl ToCode for i32 {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}_i32", self))
    }
}

impl ToCode for i64 {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}_i64", self))
    }
}

impl ToCode for u32 {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}_u32", self))
    }
}

impl ToCode for u64 {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}_u64", self))
    }
}

impl ToCode for f32 {
    fn to_code(&self) -> Result<String, String> {
        if *self == f32::NEG_INFINITY {
            Ok("f32::NEG_INFINITY".to_string())
        } else if *self == f32::INFINITY {
            Ok("f32::INFINITY".to_string())
        } else if *self == f32::NAN {
            Ok("f32::NAN".to_string())
        } else {
            Ok(format!("{}_f32", self))
        }
    }
}


impl ToCode for f64 {
    fn to_code(&self) -> Result<String, String> {
        if *self == f64::NEG_INFINITY {
            Ok("f64::NEG_INFINITY".to_string())
        } else if *self == f64::INFINITY {
            Ok("f64::INFINITY".to_string())
        } else if *self == f64::NAN {
            Ok("f64::NAN".to_string())
        } else {
            Ok(format!("{}_f64", self))
        }
    }
}

impl ToCode for bool {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("{}", self))
    }
}

impl ToCode for str {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("\"{}\"", self))
    }
}

impl ToCode for String {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("\"{}\".to_string()", self))
    }
}

impl ToCode for ResourceHandleProto {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("ResourceHandle::new({}, {}, {}, {}, {})",
                   self.get_device().to_code()?,
                   self.get_container().to_code()?,
                   self.get_name().to_code()?,
                   self.get_hash_code().to_code()?,
                   self.get_maybe_type_name().to_code()?))
    }
}

impl ToCode for VariantTensorDataProto {
    fn to_code(&self) -> Result<String, String> {
        Ok(format!("VariantTensorData::new({}, {}, {})",
                    self.get_type_name().to_code()?,
                    self.get_metadata().to_code()?,
                    self.get_tensors().to_code()?)) 
    }
}

macro_rules! check_tensor_content {
    ($member_name:ident, $tensor:ident, $found_content:ident, $rtn:ident) => (
        if !$tensor.$member_name().is_empty() {
            if $found_content {
                return Err("multiple content types".to_string());
            }
            $found_content = true;
            $rtn = $tensor.$member_name().to_code()?;
        }
    )
}

impl ToCode for TensorProto {
    fn to_code(&self) -> Result<String, String> {
        let mut found_content = false;
        let mut rtn = "".to_string();

        check_tensor_content!(get_tensor_content, self, found_content, rtn);
        check_tensor_content!(get_half_val, self, found_content, rtn);
        check_tensor_content!(get_float_val, self, found_content, rtn);
        check_tensor_content!(get_double_val, self, found_content, rtn);
        check_tensor_content!(get_int_val, self, found_content, rtn);
        check_tensor_content!(get_string_val, self, found_content, rtn);
        check_tensor_content!(get_scomplex_val, self, found_content, rtn);
        check_tensor_content!(get_int64_val, self, found_content, rtn);
        check_tensor_content!(get_bool_val, self, found_content, rtn);
        check_tensor_content!(get_dcomplex_val, self, found_content, rtn);
        check_tensor_content!(get_resource_handle_val, self, found_content, rtn);
        check_tensor_content!(get_variant_val, self, found_content, rtn);
        check_tensor_content!(get_uint32_val, self, found_content, rtn);
        check_tensor_content!(get_uint64_val, self, found_content, rtn);
        if !found_content {
            Ok("{let mut tensor: Tensor<_> = vec![].into();\n\
                       Box::new(tensor)}".to_string())
        } else {
            Ok(format!("{{let mut tensor: Tensor<_> = {}.into();\n\
                       Box::new(tensor)}}", rtn))
        }
    }
}


impl ToCode for AttrValue_ListValue {
    fn to_code(&self) -> Result<String, String> {
        if !self.get_field_type().is_empty() {
            return Err("type list is not supported".to_string());
        }

        if !self.get_func().is_empty() {
            return Err("func is not supported".to_string());
        }

        let mut found_content = false;
        let mut rtn = "vec![]".to_string();

        check_tensor_content!(get_s, self, found_content, rtn);
        check_tensor_content!(get_i, self, found_content, rtn);
        check_tensor_content!(get_f, self, found_content, rtn);
        check_tensor_content!(get_b, self, found_content, rtn);
        check_tensor_content!(get_shape, self, found_content, rtn);
        check_tensor_content!(get_tensor, self, found_content, rtn);
        if !found_content {
            Ok("vec![]".to_string())
        } else {
            Ok(rtn)
        }

    }
}