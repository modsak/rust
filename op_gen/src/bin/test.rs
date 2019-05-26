use op_gen::make_library;
use std::fs::File;
use std::io::prelude::*;


fn main() {
  let mut file = File::create("test1.rs").unwrap();
  file.write_all(b"#![allow(dead_code)]\n").unwrap();
  file.write_all(b"#![allow(non_camel_case_types)]\n").unwrap();
  file.write_all(make_library().as_ref()).unwrap();
}