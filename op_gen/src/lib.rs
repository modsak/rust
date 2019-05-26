use tensorflow_protos::op_def::OpDef;

pub mod attr;
pub mod types;
pub mod op_list;
pub mod to_code;
pub mod utils;
pub mod arg;
pub mod op_impl;

use op_impl::{OpLib, AddToLib, OpImpl};
use utils::{escape_keyword};
use arg::{make_input_arg, make_output_arg};
use attr::make_attr;
use op_list::load_oplist;

struct Op {
    op: OpDef,
}

impl Op {
    fn new(op: OpDef) -> Self {
        Self {
            op,
        }
    }
    
    fn name(&self) -> String {
        escape_keyword(self.op.get_name())
    }
}

impl AddToLib for Op {
    fn add_to_lib(&self, lib: &mut OpLib) -> Result<(), String> {
        let mut impl_ = OpImpl::new(&self.name());
        for input_arg in self.op.get_input_arg() {
            make_input_arg(input_arg.clone())?.add_to_impl(&mut impl_, lib)?;
        }

        for (port, output_arg) in self.op.get_output_arg().into_iter().enumerate() {
            make_output_arg(output_arg.clone(), port)?.add_to_impl(&mut impl_, lib)?;
        }

        for attr in self.op.get_attr() {
            make_attr(attr.clone())?.add_to_impl(&mut impl_, lib)?;
        }
        impl_.finish(lib)?;
        Ok(())
    }
}

pub fn make_library() -> String {
    let op_list = load_oplist();
    let mut op_lib = OpLib::new();
    for op_def in op_list.get_op() {
        let op = Op::new(op_def.clone());
        if let Err(msg) = op.add_to_lib(&mut op_lib) {
            println!("Not adding op {} because {}", op.name(), msg);
        }
    }
    op_lib.string()
}