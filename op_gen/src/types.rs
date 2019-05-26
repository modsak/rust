use codegen as cg;
use std::ops::Deref;
use std::fmt;

pub struct Type {
    name: String
}

impl Type {
    pub(crate) fn new<T: Into<String>>(name: T) -> Self {
        Self {
            name: name.into(),
        }
    }

    fn is_ref(&self) -> bool {
        self.name.starts_with("&")
    }

    fn add_lifetime(&mut self, lifetime: &str) {
        if !self.is_ref() {
            return
        }

        self.name = format!("&{} {}", lifetime, self.name.trim_start_matches("&"));
    }

    pub(crate) fn option(&self) -> Type {
        Type::new(format!("Option<{}>", self.name))
    }

    pub(crate) fn turbo_fish(&self) -> Type {
        Type::new(self.name.replace("<", "::<"))
    }

    fn is_box_trait(&self) -> bool {
        return self.name == "Box<dyn AnyTensor>"
    }

    fn box_trait(&self) -> Result<String, String> {
        Ok("AnyTensor".to_string())
    }
}

impl Into<cg::Type> for Type {
    fn into(self) -> cg::Type {
        self.name.into()
    }
}

impl Into<cg::Type> for &Type {
    fn into(self) -> cg::Type {
        (&self.name).into()
    }
}

impl From<cg::Type> for Type {
    fn from(ty: cg::Type) -> Self {
        (&ty).into()
    }
}

impl From<&cg::Type> for Type {
    fn from(ty: &cg::Type) -> Self {
        let mut type_name = String::new();
        {
            let mut fmt = cg::Formatter::new(&mut type_name);
            ty.fmt(&mut fmt);
        }
        Type::new(type_name)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}