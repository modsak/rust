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
               .line(format!("let mut new_op = graph.new_operation(\"{}\", &op_name)?;", &self.op_name));

        for block in &self.op_description_setup {
            make_op.push_block(block.clone());
        }

        make_op.line("let op = new_op.finish()?;");
        make_op.line("graph.record_op(self.get_id(), op.clone());");
        make_op.line("Ok(op)");
        Ok(make_op)
    }

    /// Implement the GraphOperation trait for adding to a graph
    fn graph_operation_impl(&self) -> Result<cg::Impl, String> {
        let mut graph_operation_impl = self.impl_graph_operation.clone();
        graph_operation_impl.impl_trait("GraphOperation")
                            .new_fn("get_id")
                            .arg_ref_self()
                            .ret("usize")
                            .line("self.id_");

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

// impl Finished {
//     fn new(name: &str) -> Self {
//         Self {
//             name: name.to_string(),
//             struct_: cg::Struct::new(name),
//             impl_: cg::Impl::new(name),
//             outputs: Vec::new(),
//         }
//     }

//     fn inner_type(builder: &Builder) -> cg::Type {
//         wrap_type("Rc", builder.struct_.ty())
//     }

//     fn edge_type(output: &Function) -> Result<cg::Type, String> {
//         match output.ret {
//             Some(ref ret) => Ok(ret.clone()),
//             None => Err("Output has no return type".to_string()),
//         }
//     }

//     fn into_edge_type(output: &Function) -> Result<cg::Type, String> {
//         Ok(wrap_type("Into", Self::edge_type(output)?))
//     }

//     /// Implement Into<Edge<T>> for the finished struct
//     fn add_into_edge(&self, output: Function) -> Result<cg::Impl, String> {
//         let mut impl_into = cg::Impl::new(&self.name);
//         impl_into.impl_trait(Self::into_edge_type(&output)?)
//                  .new_fn("into")
//                  .ret(Self::edge_type(&output)?)
//                  .arg_self()
//                  .line(&format!("self.{}", output.name));
//         Ok(impl_into)
//     }

//     pub(crate) fn add_output(&mut self, function: Function) {
//         self.outputs.push(function);
//     }

//     fn generic(&mut self, ty: &str) {
//         self.struct_.generic(ty);
//         self.impl_.generic(ty);
//         self.impl_.target_generic(ty);
//     }

//     fn bound(&mut self, ty: &str, bound: &str) {
//         self.struct_.bound(ty, bound);
//         self.impl_.bound(ty, bound);
//     }

//     fn finish(mut self, scope: &mut cg::Scope, builder: &Builder) -> Result<(), String> {
//         self.impl_.new_fn("new")
//                   .ret("Self")
//                   .vis("pub")
//                   .arg("inner", Self::inner_type(builder))
//                   .line("Self{inner}");

//         self.struct_.field("inner", Self::inner_type(builder));

//         if self.outputs.len() == 1 {
//             self.add_into_edge(self.outputs[0].clone())?;
//         }

//         for output in self.outputs {
//             self.impl_.push_fn(output.into());
//         }

//         scope.push_impl(self.impl_);
//         scope.push_struct(self.struct_);
//         Ok(())
//     }
// }

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





















// pub(crate) struct OpImpl<'a> {
//     lifetime_params: AsciiIter,
//     op_struct: cg::Struct,
//     op_impl: cg::Impl,
//     op_trait_impl: cg::Impl,
//     new: cg::Function,
//     make_self: cg::Block,
//     tf_operation: cg::Function,
//     op_wrapper_struct: cg::Struct,
//     inner_field: String,
//     finished_type: String,
//     op_wrapper_impl: cg::Impl,
//     op_wrapper_new: cg::Function,
//     impl_into_edge: cg::Impl,
//     num_outputs: usize,
//     lib: &'a mut OpLib,
//     phantom_i: usize,
//     generic_i: usize,
// }

// impl<'a> OpImpl<'a> {
//     fn new(op: &Op, lib: &'a mut OpLib) -> Self {
//         let mut s = Self {
//             lifetime_params: AsciiIter::new(),
//             op_struct: cg::Struct::new(&op.builder_name()),
//             op_impl: cg::Impl::new(op.builder_name()),
//             op_trait_impl: cg::Impl::new(op.builder_name()),
//             new: cg::Function::new("new"),
//             make_self: cg::Block::new("Self"),
//             tf_operation: cg::Function::new("tf_operation"),
//             op_wrapper_struct: cg::Struct::new(&op.finised_name()),
//             inner_field: format!("Rc<{}<", op.builder_name()),
//             finished_type: format!("{}<", op.finised_name()),
//             op_wrapper_impl: cg::Impl::new(&op.finised_name()),
//             op_wrapper_new: cg::Function::new("new"),
//             impl_into_edge: cg::Impl::new(&op.finised_name()),
//             num_outputs: 0,
//             lib,
//             phantom_i: 0,
//             generic_i: 0,
//         };

//         s.op_struct.field("id_", "usize").vis("pub");
//         s.op_trait_impl.impl_trait("GraphOperation")
//                        .new_fn("get_id")
//                        .arg_ref_self()
//                        .ret("usize")
//                        .line("self.id_");

//         s.new.ret("Self").vis("pub");

//         s.make_self.line("id_: new_id(),");

//         s.default_field("op_name", Type::new("String".to_string()).option(), "None");

//         s.tf_operation.arg_ref_self()
//                       .arg("graph", format!("&mut Graph"))
//                       .ret(format!("Result<Operation>"))
//                       .line("if let Some(x) = graph.get_op_by_id(self.get_id()) {")
//                       .line("    return Ok(x);")
//                       .line("}")
//                       .line("let op_name = match &self.op_name {")
//                       .line("    Some(name) => name.clone(),")
//                       .line(format!("    None => graph.new_op_name(\"{}_{{}}\")?", op.name()))
//                       .line("};")
//                       .line(format!("let mut new_op = graph.new_operation(\"{}\", &op_name)?;", op.name()));

//         s.op_wrapper_new.ret("Self");
//         s.op_wrapper_struct.vis("pub");

//         s.op_impl.new_fn("finish")
//                  .vis("pub")
//                  .arg_self()
//                  .ret(&format!("{}>", s.finished_type))
//                  .line(&format!("{}::new(Rc::new(self))", op.finised_name()));
//         s
//     }

//     fn new_generic(&mut self) -> String {
//         self.generic_i += 1;
//         format!("__T{}", self.generic_i)
//     }

//     fn generic(&mut self, ty: &str) {
//         self.op_struct.generic(ty);
//         self.op_impl.generic(ty);
//         self.op_impl.target_generic(ty);
//         self.op_trait_impl.generic(ty);
//         self.op_trait_impl.target_generic(ty);

//         self.op_wrapper_struct.generic(ty);
//         self.op_wrapper_impl.generic(ty);
//         self.op_wrapper_impl.target_generic(ty);
//         self.impl_into_edge.generic(ty);
//         self.impl_into_edge.target_generic(ty);

//         self.inner_field.push_str(&format!("{}, ", ty));
//         self.finished_type.push_str(&format!("{}, ", ty));

//         self.op_struct.field(&format!("phantom_{}", self.phantom_i), &format!("PhantomData<{}>", ty));
//         self.make_self.line(&format!("phantom_{}: PhantomData,", self.phantom_i));
//         self.phantom_i += 1;
//     }

//     fn generic_with_bound(&mut self, ty: &str, bound: &str) {
//         self.generic(ty);
//         self.bound(ty, bound);
//     }

//     fn bound(&mut self, name: &str, bound: &str) {
//         self.op_struct.bound(name, bound);
//         self.op_impl.bound(name, bound);
//         self.op_trait_impl.bound(name, bound);
//         self.op_wrapper_struct.bound(name, bound);
//         self.op_wrapper_impl.bound(name, bound);
//         self.impl_into_edge.bound(name, bound);
//     }

//     fn field<T: Into<cg::Type>>(&mut self, name: &str, ty: T) {
//         let ty_ref: &cg::Type = &ty.into();
//         self.new.arg(name, ty_ref);
//         self.op_struct.field(name, ty_ref);
//         self.make_self.line(format!("{},", name));

//     }

//     fn default_field(&mut self, name: &str, ty: Type, default: &str) {
//         self.op_struct.field(name, &ty);
//         self.make_self.line(format!("{}: {},", name, default));
//         self.op_impl.new_fn(name).arg_mut_self()
//                                  .ret("&mut Self")
//                                  .arg(name, &ty)
//                                  .line(format!("self.{} = {};", name, name))
//                                  .line("self")
//                                  .vis("pub");
//     }

//     // fn output(&mut self, port: usize, arg: &Arg) -> Result<(), String> {
//     //     self.op_wrapper_impl.new_fn(&arg.name()?)
//     //                         .arg_ref_self()
//     //                         .ret(arg.type_name()?)
//     //                         .line(&arg.constructor(port)?)
//     //                         .vis("pub");
//     //     self.num_outputs += 1;
//     //     if self.num_outputs == 1 {
//     //         self.impl_into_edge.impl_trait(format!("Into<{}>", arg.type_name()?))
//     //                            .new_fn("into")
//     //                            .arg_self()
//     //                            .ret(arg.type_name()?)
//     //                            .line(format!("{}::new(self.inner.clone(), 0)", arg.type_name()?.turbo_fish()));
//     //     }

//     //     Ok(())
//     // }

//     fn trait_field(&mut self, name: &str, trait_: &str) {
//         let arg_generic = self.new_generic();
//         self.new.arg(name, &arg_generic);
//         self.op_struct.field(name, &format!("Box<dyn {}>", trait_));
//         self.make_self.line(format!("{}: Box::new({}),", name, name));
//         self.generic_with_bound(&arg_generic, &format!("{} + 'static", trait_));
//     }

//     fn deault_trait_field(&mut self, name: &str, trait_: &str, default: &str) {
//         let arg_generic = self.new_generic();
//         self.op_struct.field(name, &format!("Box<dyn {}>", trait_));
//         self.make_self.line(format!("{}: {},", name, default));

//         self.op_impl.new_fn(name).arg_mut_self()
//                                  .ret("&mut Self")
//                                  .arg(name, &arg_generic)
//                                  .line(format!("self.{} = {};", name, name))
//                                  .line("self")
//                                  .vis("pub")
//                                  .generic(&arg_generic)
//                                  .bound(&arg_generic, &format!("{} + 'static", trait_));
//     }
// }