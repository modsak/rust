use tensorflow_protos::op_def::OpDef_ArgDef;
use tensorflow_protos::types::DataType;

use super::types::Type;
use super::utils::{tf_data_type_to_rust, escape_keyword, wrap_type};
use super::op_impl::{AddToImpl, OpImpl, OpLib, Function};

use codegen as cg;

pub(crate) fn make_input_arg(arg: OpDef_ArgDef) -> Result<Box<AddToImpl>, String> {
    let field_type = arg.get_field_type();
    let type_attr = arg.get_type_attr();
    let type_list_attr = arg.get_type_list_attr();

    match (field_type, type_attr, type_list_attr) {
        (_, "", "") => Ok(Box::new(ConcreteInputArg::new(arg))),
        (DataType::DT_INVALID, _, "") => Ok(Box::new(GenericInputArg::new(arg))),
        (DataType::DT_INVALID, "", _) => Err("Type list not currently supported".to_string()),
        _ => return Err("Invalid type combination".to_string()),
    }
}

pub(crate) fn make_output_arg(arg: OpDef_ArgDef, port: usize) -> Result<Box<AddToImpl>, String> {
    let field_type = arg.get_field_type();
    let type_attr = arg.get_type_attr();
    let type_list_attr = arg.get_type_list_attr();

    match (field_type, type_attr, type_list_attr) {
        (_, "", "") => Ok(Box::new(ConcreteOutputArg::new(arg, port))),
        (DataType::DT_INVALID, _, "") => Ok(Box::new(GenericOutputArg::new(arg, port))),
        (DataType::DT_INVALID, "", _) => Err("Type list not currently supported".to_string()),
        _ => return Err("Invalid type combination".to_string()),
    }
}

struct GenericInputArg {
    arg: Arg,
}

impl GenericInputArg {
    fn new(arg: OpDef_ArgDef) -> Self {
        Self {
            arg: Arg::new(arg),
        }
    }
}

impl AddToImpl for GenericInputArg {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        let ty = self.arg.type_attr()?;
        impl_.generic(&ty);
        impl_.bound(&ty, "TensorType");
        impl_.bound(&ty, "Clone");
        impl_.phantom(&ty);

        let arg_generic = format!("{}_Arg", self.arg.name()?);
        impl_.generic(&arg_generic);
        impl_.bound(&arg_generic, "Clone");
        impl_.bound(&arg_generic, &self.arg.arg_bound()?);
        impl_.bound(&arg_generic, "'static");

        impl_.builder.new_fn_arg(&self.arg.name()?, &cg::Type::new(&arg_generic));
        impl_.builder.make_self.line(&format!("{},", self.arg.name()?));
        impl_.builder.struct_.field(&self.arg.name()?, &arg_generic);

        let mut setup = cg::Block::new("");
        setup.line(&format!("new_op.add_edge(&self.{})?", &self.arg.name()?));
        impl_.builder.add_op_description_setup(setup);

        Ok(())
    }
}

struct ConcreteInputArg {
    arg: Arg,
}

impl ConcreteInputArg {
    fn new(arg: OpDef_ArgDef) -> Self {
        Self {
            arg: Arg::new(arg),
        }
    }
}

impl AddToImpl for ConcreteInputArg {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        let arg_generic = format!("{}_Arg", self.arg.name()?);
        impl_.generic(&arg_generic);
        impl_.bound(&arg_generic, "Clone");
        impl_.bound(&arg_generic, &self.arg.arg_bound()?);
        impl_.bound(&arg_generic, "'static");

        impl_.builder.new_fn_arg(&self.arg.name()?, &cg::Type::new(&arg_generic));
        impl_.builder.make_self.line(&format!("{},", self.arg.name()?));
        impl_.builder.struct_.field(&self.arg.name()?, &arg_generic);

        let mut setup = cg::Block::new("");
        setup.line(&format!("new_op.add_edge(&self.{})?", &self.arg.name()?));
        impl_.builder.add_op_description_setup(setup);
        Ok(())
    }
}


struct ConcreteOutputArg {
    arg: Arg,
    port: usize,
}

impl ConcreteOutputArg {
    fn new(arg: OpDef_ArgDef, port: usize) -> Self {
        Self {
            arg: Arg::new(arg),
            port,
        }
    }
}

impl AddToImpl for ConcreteOutputArg {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        let mut output_block = cg::Block::new("");
        output_block.line(format!("{}::new(rc.clone(), {})", self.arg.output_edge_type()?.turbo_fish(), self.port));
        impl_.add_output(self.arg.output_edge_type()?.into(), output_block, &self.arg.name()?);
        Ok(())
    }
}

struct GenericOutputArg {
    arg: Arg,
    port: usize,
}

impl GenericOutputArg {
    fn new(arg: OpDef_ArgDef, port: usize) -> Self {
        Self {
            arg: Arg::new(arg),
            port,
        }
    }
}

impl AddToImpl for GenericOutputArg {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        let mut output_block = cg::Block::new("");
        output_block.line(format!("{}::new(rc.clone(), {})", self.arg.output_edge_type()?.turbo_fish(), self.port));
        impl_.add_output(self.arg.output_edge_type()?.into(), output_block, &self.arg.name()?);
        impl_.generic(&self.arg.type_attr()?);
        impl_.bound(&self.arg.type_attr()?, "TensorType");
        Ok(())
    }
}

pub(crate) struct Arg {
    arg: OpDef_ArgDef,
}

impl Arg {
    fn new(arg: OpDef_ArgDef) -> Self {
        Self {
            arg,
        }
    }

    fn is_list(&self) -> bool {
        let num_attr = self.arg.get_number_attr();
        return !num_attr.is_empty()
    }

    fn setter_function(&self, var_name: &str) -> Result<String, String> {
        let is_arr = !self.arg.get_number_attr().is_empty();
        if is_arr {
            Ok(format!("add_edge_list(&{})?;", var_name)) 
        } else {
            Ok(format!("add_edge(&{})?;", var_name)) 
        }
    }

    fn name(&self) -> Result<String, String> {
        Ok(escape_keyword(self.arg.get_name()))
    }

    fn type_attr(&self) -> Result<String, String> {
        Ok(escape_keyword(self.arg.get_type_attr()))
    }

    fn type_(&self) -> Result<String, String> {
        Ok(escape_keyword(&tf_data_type_to_rust(self.arg.get_field_type())?))
    }

    fn type_name(&self) -> Result<String, String> {
        if self.type_attr()? != "" {
            self.type_attr()
        } else {
            self.type_()
        }
    }

    fn output_edge_type(&self) -> Result<Type, String> {
        let wrapper: &str;
        if self.arg.get_is_ref() {
            wrapper = "RefEdge";
        } else {
            wrapper = "Edge";
        }

        Ok(wrap_type(wrapper, self.type_name()?).into())
    }

    fn arg_bound(&self) -> Result<String, String> {
        if self.arg.get_is_ref() {
            Ok(format!("GraphRefEdge<{}>", self.type_name()?))
        } else {
            Ok(format!("GraphEdge<{}>", self.type_name()?))
        }
    }
}
