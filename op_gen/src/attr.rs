use tensorflow_protos::types::DataType;
use tensorflow_protos::op_def::OpDef_AttrDef;
use tensorflow_protos::attr_value::{AttrValue, AttrValue_oneof_value};

use super::types::Type;
use super::to_code::ToCode;
use super::utils::{escape_keyword, dump_protobuf, wrap_type, type_to_string};
use super::op_impl::{AddToImpl, OpImpl, OpLib};

use codegen as cg;

pub(crate) trait AttrBootstrap {
    fn base_type(&self) -> Result<String, String>;

    fn attr(&self) -> &Attr;

    // Field type stored in the Builder
    fn builder_field_type(&self) -> Result<String, String> {
        let inner_type;
        if self.is_list() {
            inner_type = format!("Vec<{}>", self.base_type()?);
        } else {
            inner_type = self.base_type()?;
        }

        if self.has_default() {
            Ok(format!("{}", type_to_string(&wrap_type("Option", inner_type))?))
        } else {
            Ok(inner_type.to_string())
        }
    }
    fn default_setup(&self) -> Result<String, String> {
        Ok(format!("new_op.set_attr_value_proto(\"{}\", &{})?",
                 self.attr_name(),
                (&dump_protobuf(self.default_proto()?)).to_code()?))
    }
    fn setup(&self) -> Result<String, String>;
    fn builder_arg_type(&self) -> Result<String, String>;
    fn builder_block(&self) -> Result<String, String>; // This line will be wrapped in Some() and set in the builder

    fn is_list(&self) -> bool {
        self.attr().is_list()
    }

    fn escaped_name(&self) -> Result<String, String> {
        Ok(self.attr().name()?)
    }

    fn attr_name(&self) -> &str {
        self.attr().attr.get_name()
    }

    fn default_proto(&self) -> Result<&AttrValue, String> {
        if !self.has_default() {
            return Err("No default".to_string());
        }
        Ok(self.attr().attr.get_default_value())
    }

    fn has_default(&self) -> bool {
        self.attr().attr.has_default_value()
    }

    fn update_env(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        Ok(())
    }

    fn base_builder_func(&self) -> Result<cg::Function, String> {
        let mut builder_func = cg::Function::new(&self.escaped_name()?);
        builder_func.arg(&self.escaped_name()?, &self.builder_arg_type()?)
                    .line(&format!("self.{} = Some({});",
                                   self.escaped_name()?,
                                   self.builder_block()?));
        Ok(builder_func)
    }

    fn builder_func(&self) -> Result<cg::Function, String> {
        self.base_builder_func()
    }
}

impl<T: AttrBootstrap> AddToImpl for T {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {

        let mut op_description_setup = cg::Block::new("");
        impl_.builder.struct_.field(&self.escaped_name()?, &self.builder_field_type()?);

        if self.has_default() {
            impl_.builder.make_self.line(&format!("{}: None,", self.escaped_name()?));

            impl_.builder.add_builder_fn(self.builder_func()?);

            let mut block = cg::Block::new(&format!("match self.{}", self.escaped_name()?));

            block.line(&format!("None => {},", self.default_setup()?))
                 .line(&format!("Some(ref value) => {}(&value)?,", self.setup()?))
                 .after(";");

            op_description_setup.push_block(block);

        } else {
            impl_.builder.new_fn_arg(&self.escaped_name()?, &self.builder_arg_type()?.into());
            impl_.builder.make_self.line(&format!("{}: {},", 
                                                  self.escaped_name()?,
                                                  self.builder_block()?));

            op_description_setup.line(&format!("{}(&self.{})?", self.setup()?, self.escaped_name()?));
        }
        impl_.builder.add_op_description_setup(op_description_setup);
        self.update_env(impl_, lib)?;
        Ok(())
    }
}

struct StringAttr {
    attr: Attr,
}

impl AttrBootstrap for StringAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("String".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs| {{new_op.set_attr_string_list(\"{}\", attrs)}})",
                        self.attr_name()))
        } else {
            Ok(format!("(|attr| {{new_op.set_attr_string(\"{}\", attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[String]".to_string())
        } else {
            Ok("&str".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}.to_string()", self.escaped_name()?))
        }
    }
}

struct IntAttr {
    attr: Attr,
}

impl AttrBootstrap for IntAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("i64".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs| {{new_op.set_attr_int_list(\"{}\", attrs)}})",
                        self.attr_name()))
        } else {
            Ok(format!("(|attr: &i64| {{new_op.set_attr_int(\"{}\", *attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[i64]".to_string())
        } else {
            Ok("i64".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}", self.escaped_name()?))
        }
    }
}

struct FloatAttr {
    attr: Attr,
}

impl AttrBootstrap for FloatAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("f32".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs| {{new_op.set_attr_float_list(\"{}\", attrs)}})",
                        self.attr_name()))
        } else {
            Ok(format!("(|attr: &f32| {{new_op.set_attr_float(\"{}\", *attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[f32]".to_string())
        } else {
            Ok("f32".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}", self.escaped_name()?))
        }
    }
}

struct BoolAttr {
    attr: Attr,
}

impl AttrBootstrap for BoolAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("bool".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs| {{new_op.set_attr_bool_list(\"{}\", attrs)}})",
                        self.attr_name()))
        } else {
            Ok(format!("(|attr: &bool| {{new_op.set_attr_bool(\"{}\", *attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[bool]".to_string())
        } else {
            Ok("bool".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}", self.escaped_name()?))
        }
    }
}

struct ShapeAttr {
    attr: Attr,
}

impl AttrBootstrap for ShapeAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("OtherShape".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs| {{new_op.set_attr_shape_list(\"{}\", attrs)}})",
                        self.attr_name()))
        } else {
            Ok(format!("(|attr| {{new_op.set_attr_shape(\"{}\", attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[OtherShape]".to_string())
        } else {
            Ok("&OtherShape".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}.clone()", self.escaped_name()?))
        }
    }
}

struct TensorAttr {
    attr: Attr,
}

impl TensorAttr {
    fn generic_name(&self) -> Result<String, String> {
        Ok(format!("{}_T", self.escaped_name()?))
    }

    fn tensor_type(&self) -> Result<String, String> {
        Ok(format!("{}_TensorType", self.escaped_name()?))   
    }
}

impl AttrBootstrap for TensorAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("Rc<AnyTensor>".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("(|attrs: &{}| {{new_op.set_attr_tensor_list_owned(\"{}\", attrs.clone())}})",
                        self.base_type()?,
                        self.attr_name()))
        } else {
            Ok(format!("(|attr: &{}| {{new_op.set_attr_tensor_owned(\"{}\", attr.clone())}})",
                        self.base_type()?,
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("Vec<{}>", self.generic_name()?))
        } else {
            self.generic_name()
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        Ok(format!("Rc::new(Tensor::<{}>::from({}))", self.tensor_type()?, self.escaped_name()?))
    }

    fn update_env(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        if !self.has_default() {
            let tensor_type = self.tensor_type()?;

            impl_.builder.new_fn.generic(&self.generic_name()?);
            impl_.builder.new_fn.bound(&format!("Tensor<{}>", &self.tensor_type()?),
                                       &format!("From<{}>", self.generic_name()?));
            impl_.builder.new_fn.bound(&self.generic_name()?, "'static");

            impl_.builder.direct_new_fn.generic(&self.generic_name()?);
            impl_.builder.direct_new_fn.bound(&format!("Tensor<{}>", &self.tensor_type()?),
                                              &format!("From<{}>", self.generic_name()?));
            impl_.builder.direct_new_fn.bound(&self.generic_name()?, "'static");

            impl_.builder.new_fn.generic(&tensor_type);
            impl_.builder.new_fn.bound(&tensor_type, "TensorType");

            impl_.builder.direct_new_fn.generic(&tensor_type);
            impl_.builder.direct_new_fn.bound(&tensor_type, "TensorType");
        }
        Ok(())
    }

    fn builder_func(&self) -> Result<cg::Function, String> {
        let mut func = self.base_builder_func()?;
        func.generic(&self.generic_name()?)
            .bound(&format!("Tensor<{}>", &self.tensor_type()?),
                   &format!("From<{}>", self.generic_name()?))
            .bound(&self.generic_name()?, "'static")
            .generic(&self.tensor_type()?)
            .bound(&self.tensor_type()?, "TensorType");
        Ok(func)
    }
}

struct FuncAttr {
    attr: Attr,
}

impl AttrBootstrap for FuncAttr {
    fn attr(&self) -> &Attr {
        &self.attr
    }

    fn base_type(&self) -> Result<String, String> {
        Ok("String".to_string())
    }

    fn setup(&self) -> Result<String, String> {
        if self.is_list() {
            Err("Function list attr not supported".to_string())
        } else {
            Ok(format!("(|attr| {{new_op.set_attr_func_name(\"{}\", attr)}})",
                        self.attr_name()))
        }
    }

    fn builder_arg_type(&self) -> Result<String, String> {
        if self.is_list() {
            Ok("&[String]".to_string())
        } else {
            Ok("&str".to_string())
        }
    }

    fn builder_block(&self) -> Result<String, String> {
        if self.is_list() {
            Ok(format!("{}.to_vec()", self.escaped_name()?))
        } else {
            Ok(format!("{}.to_string()", self.escaped_name()?))
        }
    }
}


struct TypeAttr {
    attr: Attr,
}

impl AddToImpl for TypeAttr {
    fn add_to_impl(&self, impl_: &mut OpImpl, lib: &mut OpLib) -> Result<(), String> {
        let ty = self.attr.name()?;
        impl_.generic(&ty);
        if self.attr.allowed_types().len() > 0 {
            let bound = lib.contraint_trait(self.attr.allowed_types())?;
            impl_.bound(&ty, &bound);
        }

        impl_.bound(&ty, "TensorType");
        impl_.bound(&ty, "'static");
        impl_.bound(&ty, "Clone");
        impl_.phantom(&ty);

        let mut op_setup = cg::Block::new("");
        op_setup.line(&format!("new_op.set_attr_type(\"{}\", {}::data_type())?;", &ty, &ty));
        impl_.builder.add_op_description_setup(op_setup);
        Ok(())
    }
}


pub(crate) fn make_attr(attr: OpDef_AttrDef) -> Result<Box<dyn AddToImpl>, String> {
    let attr = Attr::new(attr);
    match attr.bare_type_name() {
        "string" => Ok(Box::new(StringAttr {attr})),
        "float" => Ok(Box::new(FloatAttr {attr})),
        "int" => Ok(Box::new(IntAttr {attr})),
        "bool" => Ok(Box::new(BoolAttr {attr})),
        "shape" => Ok(Box::new(ShapeAttr {attr})),
        "tensor" => Ok(Box::new(TensorAttr {attr})),
        "func" => Ok(Box::new(FuncAttr {attr})),
        "type" => Ok(Box::new(TypeAttr {attr})),
        "placeholder" => Err("Placeholder attrs are not supported".to_string()),
        ty => Err(format!("Unknown type {}", ty)),
    }
}

#[derive(Clone)]
pub(crate) struct Attr {
    attr: OpDef_AttrDef,
}

impl Attr {
    fn new(attr: OpDef_AttrDef) -> Self {
        Self {
            attr,
        }
    }

    fn default(&self) -> Result<String, String> {
        use AttrValue_oneof_value::*;

        if !self.has_default_value() {
            return Err("No default value".to_string());
        }

        let default_value = &self.attr.get_default_value().value;
        for v in default_value {
            let r = match v {
                s(string_vec) => Ok(format!("String::from_utf8_lossy(&{}).to_string()", string_vec.to_code()?)),
                i(int) => int.to_code(),
                f(float) => float.to_code(),
                b(bo) => bo.to_code(),
                field_type(_) => return Err("No support for default type_fields".to_string()),
                shape(shape_proto) => shape_proto.to_code(),
                tensor(tensor_proto) => tensor_proto.to_code(),
                list(list_value) => list_value.to_code(),
                func(_) => return Err("func not supported".to_string()),
                placeholder(placeholder_str) => placeholder_str.to_code(),
            };
            return r;
        }
        return Err("No default value".to_string());

    }

    fn has_default_value(&self) -> bool {
        self.attr.has_default_value()
    }

    fn bare_type_name(&self) -> &str {
        if self.is_list() {
            let end = self.attr.get_field_type().len();
            &self.attr.get_field_type()[5..end-1]
        } else {
            self.attr.get_field_type()
        }
    }

    fn is_list(&self) -> bool {
        self.attr.get_field_type().starts_with("list(") && self.attr.get_field_type().ends_with(")")
    }

    fn type_name(&self) -> Result<Type, String> {
        let bare_type_name = match self.bare_type_name() {
            "string" => "String",
            "float" => "f32",
            "int" => "i64",
            "shape" => "OtherShape",
            "tensor" => "Box<dyn AnyTensor>",
            "func" => return Err("func attrs not supported".to_string()),
            s => s,
        };

        if self.is_list() {
            Ok(Type::new( format!("Vec<{}>", bare_type_name)))
        } else {
            Ok(Type::new(escape_keyword(&bare_type_name.to_string())))
        }
    }

    fn setter_function(&self, var_name: &str) -> Result<String, String> {
        Ok(match self.attr.get_field_type() {
            "string" => format!("set_attr_string(\"{}\", &{})?", self.name()?, var_name),
            "list(string)" => format!("set_attr_string_list(\"{}\", &{})?", self.name()?, var_name),
            "int" => format!("set_attr_int(\"{}\", {})?", self.name()?, var_name),
            "list(int)" => format!("set_attr_int_list(\"{}\", &{})?", self.name()?, var_name),
            "float" => format!("set_attr_float(\"{}\", {})?", self.name()?, var_name),
            "list(float)" => format!("set_attr_float_list(\"{}\", &{})?", self.name()?, var_name),
            "bool" => format!("set_attr_bool(\"{}\", {})?", self.name()?, var_name),
            "list(bool)" => format!("set_attr_bool_list(\"{}\", &{})?", self.name()?, var_name),
            "shape" => format!("set_attr_shape(\"{}\", &{})?", self.name()?, var_name),
            "list(shape)" => format!("set_attr_shape_list(\"{}\", &{})?", self.name()?, var_name),
            "tensor" => format!("set_attr_tensor(\"{}\", {}.clone())?", self.name()?, var_name),
            "list(tensor)" => format!("set_attr_tensor_list(\"{}\", {}.clone())?", self.name()?, var_name),
            s => return Err(format!("Attr type {} not supported", s)),
        })
    }

    fn allowed_types(&self) -> Vec<DataType> {
        if !self.attr.has_allowed_values() {
            return vec![];
        }

        let attr_value = self.attr.get_allowed_values();

        if !attr_value.has_list() {
            return vec![];
        }

        attr_value.get_list().get_field_type().to_vec()
    }

    fn name(&self) -> Result<String, String> {
        Ok(escape_keyword(self.attr.get_name()))
    }
}
