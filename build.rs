use op_gen::make_library;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::io::prelude::*;

fn recurse_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in dir.read_dir().unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            files.push(entry.path());
        } else {
            files.extend(recurse_files(&entry.path()));
        }
    }
    files
}

fn main() {
  
  for file in recurse_files(&PathBuf::from("op_gen")) {
    println!("cargo:rerun-if-changed={}", file.to_str().unwrap());
  }

  for file in recurse_files(&PathBuf::from("tensorflow-protos")) {
    println!("cargo:rerun-if-changed={}", file.to_str().unwrap());
  }

  let mut file = File::create("src/ops.rs").unwrap();
  file.write_all(b"#![allow(missing_docs)]\n").unwrap();
  file.write_all(b"#![allow(non_camel_case_types)]\n").unwrap();
  file.write_all(b"#![allow(non_snake_case)]\n").unwrap();
  file.write_all(b"#![allow(non_upper_case_globals)]\n").unwrap();
  file.write_all(b"#![allow(dead_code)]\n").unwrap();
  file.write_all(b"#![allow(non_camel_case_types)]\n").unwrap();
  file.write_all(b"#![allow(missing_debug_implementations)]\n").unwrap();
  file.write_all(b"#![allow(unused_variables)]\n").unwrap();
  file.write_all(b"#![allow(unused_mut)]\n").unwrap();
  
  file.write_all(make_library().as_ref()).unwrap();
}