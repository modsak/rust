extern crate protobuf_codegen_pure;


fn main() {
  let files = vec!["protos/tensorflow/core/framework/op_def.proto",
               "protos/tensorflow/core/framework/attr_value.proto",
               "protos/tensorflow/core/framework/types.proto",
               "protos/tensorflow/core/framework/tensor_shape.proto",
               "protos/tensorflow/core/framework/tensor.proto",
               "protos/tensorflow/core/framework/resource_handle.proto"];

  for file in &files {
    println!("cargo:rerun-if-changed={}", file);
  }

  protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
      out_dir: "src",
      input: &files,
      includes: &["protos"],
      customize: protobuf_codegen_pure::Customize {
        ..Default::default()
      },
  }).expect("protoc");
}