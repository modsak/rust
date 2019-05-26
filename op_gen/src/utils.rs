use std::iter::{Cycle, Peekable};
use tensorflow_protos::types::DataType;
use std::fmt::Display;
use codegen as cg;
use protobuf;

pub(crate) fn escape_keyword(name: &str) -> String {
    match name {
        "as" => format!("{}_", name),
        "break" => format!("{}_", name),
        "const" => format!("{}_", name),
        "continue" => format!("{}_", name),
        "crate" => format!("{}_", name),
        "else" => format!("{}_", name),
        "enum" => format!("{}_", name),
        "extern" => format!("{}_", name),
        "false" => format!("{}_", name),
        "fn" => format!("{}_", name),
        "for" => format!("{}_", name),
        "if" => format!("{}_", name),
        "impl" => format!("{}_", name),
        "in" => format!("{}_", name),
        "let" => format!("{}_", name),
        "loop" => format!("{}_", name),
        "match" => format!("{}_", name),
        "mod" => format!("{}_", name),
        "move" => format!("{}_", name),
        "mut" => format!("{}_", name),
        "pub" => format!("{}_", name),
        "ref" => format!("{}_", name),
        "return" => format!("{}_", name),
        "self" => format!("{}_", name),
        "Self" => format!("{}_", name),
        "static" => format!("{}_", name),
        "struct" => format!("{}_", name),
        "super" => format!("{}_", name),
        "trait" => format!("{}_", name),
        "true" => format!("{}_", name),
        "type" => format!("{}_", name),
        "unsafe" => format!("{}_", name),
        "use" => format!("{}_", name),
        "where" => format!("{}_", name),
        "while" => format!("{}_", name),
        "dyn" => format!("{}_", name),
        "abstract" => format!("{}_", name),
        "become" => format!("{}_", name),
        "box" => format!("{}_", name),
        "do" => format!("{}_", name),
        "final" => format!("{}_", name),
        "macro" => format!("{}_", name),
        "override" => format!("{}_", name),
        "priv" => format!("{}_", name),
        "typeof" => format!("{}_", name),
        "unsized" => format!("{}_", name),
        "virtual" => format!("{}_", name),
        "yield" => format!("{}_", name),
        "async" => format!("{}_", name),
        "await" => format!("{}_", name),
        "try" => format!("{}_", name),
        "union" => format!("{}_", name),
        _ => name.to_string(),
    }

}

fn single_ascii_iter() -> Box<dyn Iterator<Item=char>> {
    Box::new((0..26).map(|x| (x + b'a') as char))
}

struct SingleAsciiIter {
    iter: Box<dyn Iterator<Item=char>>,
}

impl SingleAsciiIter {
    fn new() -> Self {
        Self {
            iter: single_ascii_iter(),
        }
    }
}

impl Iterator for SingleAsciiIter {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        self.iter.next()
    }
}

impl Clone for SingleAsciiIter {
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// Infinite iterator that produces unique strings of lower case ascii. e.g (a, b, c ... aa, ab, ac)
pub(crate) struct AsciiIter {
    iters: Vec<Peekable<Cycle<SingleAsciiIter>>>,
    pos: Vec<u8>,
}

impl AsciiIter {
    pub(crate) fn new() -> Self {
        Self {
            iters: Vec::new(),
            pos: Vec::new(),
        }
    }

    fn add_char(&mut self) {
        self.iters.push(SingleAsciiIter::new().cycle().peekable());
        self.pos.push(0);
    }
}

impl Iterator for AsciiIter {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        for (idx, i) in self.pos.iter_mut().enumerate() {
            self.iters[idx].next();
            *i += 1;
            *i %= 26;

            if *i != 0 {
                return Some(self.iters.iter_mut().map(|x| x.peek().unwrap()).collect());
            }
        }

        self.add_char();
        Some(self.iters.iter_mut().map(|x| x.peek().unwrap()).collect())
    }
}


pub(crate) fn tf_data_type_to_rust(data_type: DataType) -> Result<String, String> {
    match data_type {
        DataType::DT_FLOAT => Ok("f32".to_string()),
        DataType::DT_DOUBLE => Ok("f64".to_string()),
        DataType::DT_INT32 => Ok("i32".to_string()),
        DataType::DT_UINT8 => Ok("u8".to_string()),
        DataType::DT_INT16 => Ok("i16".to_string()),
        DataType::DT_INT8 => Ok("i8".to_string()),
        DataType::DT_STRING => Ok("String".to_string()),
        DataType::DT_COMPLEX64 => Ok("OtherComplex<f32>".to_string()),
        DataType::DT_INT64 => Ok("i64".to_string()),
        DataType::DT_BOOL => Ok("bool".to_string()),
        DataType::DT_QINT8 => Ok("i8".to_string()),
        DataType::DT_QUINT8 => Ok("u8".to_string()),
        DataType::DT_QINT32 => Ok("i32".to_string()),
        DataType::DT_BFLOAT16 => Ok("BFloat16".to_string()),
        DataType::DT_QINT16 => Ok("i16".to_string()),
        DataType::DT_QUINT16 => Ok("u16".to_string()),
        DataType::DT_UINT16 => Ok("u16".to_string()),
        DataType::DT_COMPLEX128 => Ok("OtherComplex<f64>".to_string()),
        DataType::DT_UINT32 => Ok("u32".to_string()),
        DataType::DT_UINT64 => Ok("u64".to_string()),
        _ => Err(format!("Unsupported data type {:?}", data_type))
    }
}

pub(crate) fn wrap_type<T: Into<cg::Type>, U: Into<cg::Type>>(wrapper: T, inner: U) -> cg::Type {
    let mut wrapped: cg::Type = wrapper.into();
    wrapped.generic(inner);
    wrapped
}

pub(crate) fn type_to_string(ty: &cg::Type) -> Result<String, String> {
    let mut s = String::new();
    let mut formatter = cg::Formatter::new(&mut s);
    if let Err(_) = ty.fmt(&mut formatter) {
        Err("Failed to format".to_string())
    } else {
        Ok(s)
    }
}

pub(crate) fn dump_protobuf<T: protobuf::Message>(message: &T) -> Vec<u8>{
    let mut buffer = Vec::new();
    message.write_to_with_cached_sizes(&mut protobuf::CodedOutputStream::vec(&mut buffer)).expect("Failed to dump protobuf");
    buffer
}

pub(crate) fn join_vec<T>(v: &Vec<T>, delim: &str) -> String
where for<'a> &'a T: Display {
    let mut rtn = String::new();
    if v.len() == 0 {
        return "".to_string();
    }

    for elem in v.into_iter().take(v.len() - 1) {
        rtn.push_str(&format!("{}{}", elem, delim));
    }

    rtn.push_str(&format!("{}", &v[v.len()-1]));
    rtn
}