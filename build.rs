use op_gen::make_library;
use std::fs::File;
use std::io::prelude::*;


fn main() {
  let mut file = File::create("src/ops.rs").unwrap();
  file.write_all(b"#![allow(missing_docs)]\n").unwrap();
  file.write_all(b"#![allow(non_camel_case_types)]\n").unwrap();
  file.write_all(b"#![allow(non_snake_case)]\n").unwrap();
  file.write_all(b"#![allow(non_upper_case_globals)]\n").unwrap();
  file.write_all(b"#![allow(dead_code)]\n").unwrap();
  file.write_all(b"#![allow(non_camel_case_types)]\n").unwrap();
  file.write_all(make_library().as_ref()).unwrap();
}