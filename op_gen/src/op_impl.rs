use std::ops;
use std::collections::{HashMap, HashSet};
use super::utils::{tf_data_type_to_rust, wrap_type, join_vec, type_to_string};
use tensorflow_protos::types::DataType;

use codegen as cg;

pub(crate) trait AddToImpl {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String>;
}

pub(crate) trait AddToLib {
    fn add_to_lib(&self, lib: &mut OpLib) -> Result<(), String>;
}


pub(crate) struct OpLib {
    // Maps sets of allowed types to traits that constrain to those types
    type_constraint_traits: HashMap<Vec<DataType>, String>,
    scope: cg::Scope,
}

impl OpLib {
    pub(crate) fn new() -> Self {
        let mut lib = Self {
            type_constraint_traits: HashMap::new(),
            scope: cg::Scope::new(),
        };
        lib.scope.import("super::graph", "Graph");
        lib.scope.import("super::graph", "GraphOperation");
        lib.scope.import("super::graph", "Operation");
        lib.scope.import("super::graph", "Edge");
        lib.scope.import("super::graph", "RefEdge");
        lib.scope.import("super::graph", "GraphEdge");
        lib.scope.import("super::graph", "GraphRefEdge");
        lib.scope.import("super", "Shape as OtherShape");
        lib.scope.import("super", "new_id");
        lib.scope.import("super", "TensorType");
        lib.scope.import("super", "BFloat16");
        lib.scope.import("super", "Tensor");
        lib.scope.import("super", "AnyTensor");
        lib.scope.import("num_complex", "Complex as OtherComplex");
        lib.scope.import("std::rc", "Rc");
        lib.scope.import("std", "f32");
        lib.scope.import("std", "f64");
        lib.scope.import("std::convert", "From");
        lib.scope.import("std::marker", "PhantomData");
        lib.scope.import("super", "Result");
        lib
    }

    fn contraint_name(allowed_types: &Vec<DataType>) -> String {
        allowed_types.iter().fold("con".to_string(), |acc, x| format!("{}_or_{:?}", acc, x))
    }

    pub(crate) fn contraint_trait(&mut self, mut allowed_types: Vec<DataType>) -> Result<String, String> {
        allowed_types.sort_by_key(|a| *a as u32);

        if let Some(x) = self.type_constraint_traits.get(&allowed_types) {
            return Ok(x.to_string());
        }

        let new_constraint_name = OpLib::contraint_name(&allowed_types);
        self.scope.new_trait(&new_constraint_name).vis("pub");

        let mut dtypes: Vec<String> = allowed_types.iter()
                                      .filter_map(|x| {
                                            if let Ok(x) = tf_data_type_to_rust(*x) {
                                                Some(x)
                                            } else {
                                                None
                                            }
                                        })
                                      .collect();
        dtypes.sort();
        dtypes.dedup();

        for dtype in dtypes {
            self.scope.new_impl(&dtype)
                      .impl_trait(&new_constraint_name);
        }
        self.type_constraint_traits.insert(allowed_types, new_constraint_name.clone());
        Ok(new_constraint_name)

    }

    pub(crate) fn string(&self) -> String {
        self.scope.to_string()
    }
}


/// Represents a struct for building an operation, builders must have static lifetime
/// becuase they get put in an RC, so no lifetime parameters
pub(crate) struct Builder {
    name: String,
    pub(crate) struct_: cg::Struct,
    pub(crate) impl_: cg::Impl,
    pub(crate) impl_graph_operation: cg::Impl,
    pub(crate) new_fn: cg::Function,
    pub(crate) direct_new_fn: cg::Function,
    pub(crate) make_self: cg::Block,
    pub(crate) op_description_setup: Vec<cg::Block>,
    outputs: Vec<(cg::Type, cg::Block, String)>,

    // Name used by tensorflow
    op_name: String, 
}

/// Wrapper around cg::Function as cg::Function does not allow access to the return type once
/// it's been set
#[derive(Clone)]
pub(crate) struct Function {
    func: cg::Function,
    name: String,
    ret: Option<cg::Type>,
}

impl ops::Deref for Function {
    type Target = cg::Function;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl ops::DerefMut for Function {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.func
    }
}

impl Into<cg::Function> for Function {
    fn into(self) -> cg::Function {
        self.func
    }
}

impl AsRef<cg::Function> for Function {
    fn as_ref(&self) -> &cg::Function {
        &self.func
    }
}

impl Function {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            func: cg::Function::new(name),
            name: name.to_string(),
            ret: None,
        }
    }

    pub(crate) fn ret<T: Into<cg::Type>>(&mut self, ret: T) -> &mut Self {
        let ret_copy = ret.into();
        self.func.ret(&ret_copy);
        self.ret = Some(ret_copy);
        self
    }

    fn get_ret(&self) -> cg::Type {
        match self.ret {
            Some(ref ty) => ty.clone(),
            None => cg::Type::new("()"),
        }
    }
}

/// Represents a struct for using the operation once it's added to a graph
// #[derive(Clone)]
// pub(crate) struct Finished {
//     name: String,
//     struct_: cg::Struct,
//     impl_: cg::Impl,
//     outputs: Vec<Function>,
// }

pub(crate) struct OpImpl {
    pub(crate) builder: Builder,
    generics: HashSet<String>,
    phantoms: HashSet<String>,
    bounds: HashSet<(String, String)>,
}


impl Builder {
    fn new(name: &str, op_name: &str) -> Self {
        Self {
            name: name.to_string(),
            struct_: cg::Struct::new(name),
            impl_: cg::Impl::new(name),
            impl_graph_operation: cg::Impl::new(name),
            new_fn: cg::Function::new("build"),
            direct_new_fn: cg::Function::new("new"),
            make_self: cg::Block::new("Self"),
            op_description_setup: Vec::new(),
            op_name: op_name.to_string(),
            outputs: Vec::new(),
        }
    }

    /// Do the operation description setup to add the operation to a graph
    fn make_op(&self) -> Result<cg::Function, String> {
        let mut make_op = cg::Function::new("tf_operation");

        make_op.arg_ref_self()
               .arg("graph", format!("&mut Graph"))
               .ret(format!("Result<Operation>"))
               .line("if let Some(x) = graph.get_op_by_id(self.get_id()) {")
               .line("    return Ok(x);")
               .line("}")
               .line("let op_name = match &self.op_name {")
               .line("    Some(name) => name.clone(),")
               .line(format!("    None => graph.new_op_name(\"{}_{{}}\")?", &self.op_name))
               .line("};")
               .line("let mut control_inputs = Vec::new();")
               .line("for control_input in self.control_inputs.iter() {")
               .line("    control_inputs.push(control_input.tf_operation(graph)?);")
               .line("}")
               .line(format!("let mut new_op = graph.new_operation(\"{}\", &op_name)?;", &self.op_name))
               .line("for control_input in control_inputs {")
               .line("    new_op.add_control_input(&control_input)")
               .line("}");


        for block in &self.op_description_setup {
            make_op.push_block(block.clone());
        }

        make_op.line("let op = new_op.finish()?;");
        make_op.line("graph.record_op(self.get_id(), op.clone());");
        make_op.line("Ok(op)");
        Ok(make_op)
    }

    /// Implement the GraphOperation trait for adding to a graph
    fn graph_operation_impl(&mut self) -> Result<cg::Impl, String> {
        self.impl_.new_fn("get_id")
                  .arg_ref_self()
                  .ret("usize")
                  .line("self.id_");

        let mut graph_operation_impl = self.impl_graph_operation.clone();
        graph_operation_impl.impl_trait("GraphOperation");
        graph_operation_impl.push_fn(self.make_op()?);
        Ok(graph_operation_impl)
    }

    fn finish_ret(&self) -> cg::Type {
        if self.outputs.len() == 1 {
            return self.outputs[0].0.clone();
        }

        let rets = self.outputs.iter().map(|x| type_to_string(&x.0).unwrap()).collect();
        format!("({})", join_vec(&rets, ", ")).into()
    }

    /// Make the finish function for turning the builder into the real op
    fn make_finish(&self) -> Result<cg::Function, String> {
        let mut make_finish = cg::Function::new("finish");

        make_finish.ret(self.finish_ret())
                   .arg_self()
                   .vis("pub")
                   .line("let rc = Rc::new(self);");


        if self.outputs.len() != 1 {
            make_finish.line("(");
            for output in &self.outputs {
                let mut blk = output.1.clone();
                blk.after(",");
                make_finish.push_block(blk);
            }
            make_finish.line(")");
        } else {
            make_finish.push_block(self.outputs[0].1.clone());
        }

        Ok(make_finish)
    }

    /// Allow the user to give the op a meaningful name
    fn add_optional_name(&mut self) {
        self.struct_.field("op_name", "Option<String>");
        self.make_self.line("op_name: None,");
        self.impl_.new_fn("op_name")
                  .vis("pub")
                  .arg_mut_self()
                  .arg("op_name", "&str")
                  .ret("&mut Self")
                  .line("self.op_name = Some(op_name.to_string());")
                  .line("self");
    }

    fn add_control_inputs(&mut self) {
        self.struct_.field("control_inputs", "Vec<Rc<dyn GraphOperation>>");
        self.make_self.line("control_inputs: Vec::new(),");
        self.impl_.new_fn("control_input")
                  .vis("pub")
                  .arg_mut_self()
                  .generic("ControlInputT")
                  .bound("ControlInputT", "GraphOperation")
                  .bound("ControlInputT", "Clone")
                  .bound("ControlInputT", "'static")
                  .arg("control_input", "ControlInputT")
                  .ret("&mut Self")
                  .line("self.control_inputs.push(Rc::new(control_input.clone()));")
                  .line("self");
    }

    fn generic(&mut self, ty: &str) {
        self.struct_.generic(ty);
        self.impl_.generic(ty);
        self.impl_.target_generic(ty);
        self.impl_graph_operation.generic(ty);
        self.impl_graph_operation.target_generic(ty);
    }

    fn bound(&mut self, ty: &str, bound: &str) {
        self.struct_.bound(ty, bound);
        self.impl_.bound(ty, bound);
        self.impl_graph_operation.bound(ty, bound);
    }

    fn phantom(&mut self, ty: &str) {
        let field_name = &format!("phantom_{}", ty);
        self.struct_.field(field_name, &format!("PhantomData<{}>", ty));
        self.make_self.line(&format!("{}: PhantomData,", field_name));
    }

    pub(crate) fn add_builder_fn(&mut self, mut func: cg::Function) {
        // Ideally would take "mut self" and avoid the clone but
        // codegen doesn't currently support that
        func.ret("Self")
            .arg_mut_self()
            .vis("pub")
            .line("self.clone()");
        self.impl_.push_fn(func);
    } 

    pub(crate) fn add_op_description_setup(&mut self, block: cg::Block) {
        self.op_description_setup.push(block);
    }

    fn add_output(&mut self, ty: cg::Type, block: cg::Block, name: &str) {
        let mut output_fn = cg::Function::new(name);
        output_fn.arg_self()
                 .vis("pub")
                 .ret((&ty).clone())
                 .line("let rc = Rc::new(self);")
                 .push_block((&block).clone());
        self.impl_.push_fn(output_fn);

        self.outputs.push((ty, block, name.to_string()));
    }

    pub(crate) fn new_fn_arg(&mut self, name: &str, ty: &cg::Type) {
        self.new_fn.arg(name, ty);
        self.direct_new_fn.arg(name, ty);
    }

    pub(crate) fn new_fn_generic(&mut self, name: &str) {
        self.new_fn.generic(name);
        self.direct_new_fn.generic(name);
    }

    pub(crate) fn new_fn_bound(&mut self, name: &str, bound: &str) {
        self.new_fn.bound(name, bound);
        self.direct_new_fn.bound(name, bound);
    }

    fn setup_direct_new(&mut self) {
        let finish_ret = self.finish_ret();
        let mut make_self = self.make_self.clone();
        make_self.after(".finish()");

        self.direct_new_fn.vis("pub")
                          .ret(finish_ret)
                          .push_block(make_self);

    }

    /// Add all the generated code to the scope
    fn finish(mut self, scope: &mut cg::Scope) -> Result<(), String> {
        self.add_optional_name();
        self.add_control_inputs();
        scope.push_impl(self.graph_operation_impl()?);
        self.impl_.push_fn(self.make_finish()?);

        self.new_fn.ret("Self")
                   .vis("pub");
        self.struct_.field("id_", "usize")
                    .vis("pub");
        self.make_self.line("id_: new_id(),");

        self.struct_.derive("Clone");
        self.setup_direct_new();

        self.new_fn.push_block(self.make_self);
        self.impl_.push_fn(self.new_fn);
        self.impl_.push_fn(self.direct_new_fn);
        scope.push_struct(self.struct_);
        scope.push_impl(self.impl_);

        Ok(())
    }
}


impl OpImpl {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            builder: Builder::new(name, name),
            generics: HashSet::new(),
            phantoms: HashSet::new(),
            bounds: HashSet::new(),
        }
    }

    pub(crate) fn finish(self, lib: &mut OpLib) -> Result<(), String> {
        self.builder.finish(&mut lib.scope)?;
        Ok(())
    }

    pub(crate) fn generic(&mut self, ty: &str) {
        if self.generics.contains(ty) {
            return;
        }
        self.generics.insert(ty.to_string());
        self.builder.generic(ty);
    }

    pub(crate) fn phantom(&mut self, ty: &str) {
        if self.phantoms.contains(ty) {
            return;
        }
        self.phantoms.insert(ty.to_string());
        self.builder.phantom(ty);   
    }

    pub(crate) fn bound(&mut self, ty: &str, bound: &str) {
        if self.bounds.contains(&(ty.to_string(), bound.to_string())) {
            return;
        }

        self.bounds.insert((ty.to_string(), bound.to_string()));
        self.builder.bound(ty, bound);
    }

    pub(crate) fn add_output(&mut self, ty: cg::Type, block: cg::Block, name: &str) {
        self.builder.add_output(ty, block, name)
    }
}

