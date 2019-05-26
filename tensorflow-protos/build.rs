extern crate protobuf_codegen_pure;


fn main() {
  protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
      out_dir: "src",
      input: &["protos/tensorflow/core/framework/op_def.proto",
               "protos/tensorflow/core/framework/attr_value.proto",
               "protos/tensorflow/core/framework/types.proto",
               "protos/tensorflow/core/framework/tensor_shape.proto",
               "protos/tensorflow/core/framework/tensor.proto",
               "protos/tensorflow/core/framework/resource_handle.proto"],
      includes: &["protos"],
      customize: protobuf_codegen_pure::Customize {
        ..Default::default()
      },
  }).expect("protoc");
}