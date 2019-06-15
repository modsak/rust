use super::{TensorType, Tensor, Result, Operation, Shape, GraphEdge, new_id};
use super::graph::{Edge, RefEdge, GraphOperation, Graph, Output};
use super::ops;
use std::ops::*;
use std::iter::repeat;
use std::rc::Rc;
use std::cell::Cell;

/// Add a constant node to the graph
/// e.g.
/// let c = constant(vec![1, 2, 3]);
pub fn constant<T: TensorType, U>(value: U) -> Edge<T>
where Tensor<T>: From<U>,
      U: 'static {
    ops::Const::new(value)
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_STRING_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Add for Edge<T> {
    type Output = Edge<T>;
    fn add(self, o: Edge<T>) -> Edge<T> {
        ops::Add::new(self, o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_STRING_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Add for &Edge<T> {
    type Output = Edge<T>;
    fn add(self, o: &Edge<T>) -> Edge<T> {
        ops::Add::new(self.clone(), o.clone())
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_STRING_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Add<Edge<T>> for &Edge<T> {
    type Output = Edge<T>;
    fn add(self, o: Edge<T>) -> Edge<T> {
        ops::Add::new(self.clone(), o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_STRING_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Add<&Edge<T>> for Edge<T> {
    type Output = Edge<T>;
    fn add(self, o: &Edge<T>) -> Edge<T> {
        ops::Add::new(self, o.clone())
    }
}



impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Sub for Edge<T> {
    type Output = Edge<T>;
    fn sub(self, o: Edge<T>) -> Edge<T> {
        ops::Sub::new(self, o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Sub for &Edge<T> {
    type Output = Edge<T>;
    fn sub(self, o: &Edge<T>) -> Edge<T> {
        ops::Sub::new(self.clone(), o.clone())
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Sub<Edge<T>> for &Edge<T> {
    type Output = Edge<T>;
    fn sub(self, o: Edge<T>) -> Edge<T> {
        ops::Sub::new(self.clone(), o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Sub<&Edge<T>> for Edge<T> {
    type Output = Edge<T>;
    fn sub(self, o: &Edge<T>) -> Edge<T> {
        ops::Sub::new(self, o.clone())
    }
}




impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Mul for Edge<T> {
    type Output = Edge<T>;
    fn mul(self, o: Edge<T>) -> Edge<T> {
        ops::Mul::new(self, o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Mul for &Edge<T> {
    type Output = Edge<T>;
    fn mul(self, o: &Edge<T>) -> Edge<T> {
        ops::Mul::new(self.clone(), o.clone())
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Mul<Edge<T>> for &Edge<T> {
    type Output = Edge<T>;
    fn mul(self, o: Edge<T>) -> Edge<T> {
        ops::Mul::new(self.clone(), o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Mul<&Edge<T>> for Edge<T> {
    type Output = Edge<T>;
    fn mul(self, o: &Edge<T>) -> Edge<T> {
        ops::Mul::new(self, o.clone())
    }
}




impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Div for Edge<T> {
    type Output = Edge<T>;
    fn div(self, o: Edge<T>) -> Edge<T> {
        ops::Div::new(self, o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Div for &Edge<T> {
    type Output = Edge<T>;
    fn div(self, o: &Edge<T>) -> Edge<T> {
        ops::Div::new(self.clone(), o.clone())
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Div<Edge<T>> for &Edge<T> {
    type Output = Edge<T>;
    fn div(self, o: Edge<T>) -> Edge<T> {
        ops::Div::new(self.clone(), o)
    }
}

impl<T: ops::con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF + TensorType> Div<&Edge<T>> for Edge<T> {
    type Output = Edge<T>;
    fn div(self, o: &Edge<T>) -> Edge<T> {
        ops::Div::new(self, o.clone())
    }
}

pub struct Variable<T: TensorType> {
    variable: ops::VariableV2<T>,
    initialiser: Box<dyn Initialiser<T>>,
    shape: Vec<u64>,
    validate_shape: bool,
}

impl<T:TensorType> Variable<T> {
    /// Returns the value and initialiser op for the variable, in that order
    pub fn new<I: Initialiser<T> + 'static>(shape: &[u64], initialiser: I) -> Result<(RefEdge<T>, RefEdge<T>)> {
        Self::build(shape, initialiser).finish()
    }

    pub fn build<I: Initialiser<T> + 'static>(shape: &[u64], initialiser: I) -> Self {
        let shape_enum = Shape(Some(shape.into_iter().map(|x| Some(*x as i64)).collect()));
        Self {
            variable: ops::VariableV2::build(&shape_enum),
            initialiser: Box::new(initialiser),
            shape: shape.to_vec(),
            validate_shape: true,
        }
    }

    pub fn validate_shape(&mut self, validate_shape: bool) -> &mut Self {
        self.validate_shape = validate_shape;
        self
    }

    /// Returns the value and initialiser op for the variable, in that order
    pub fn finish(self) -> Result<(RefEdge<T>, RefEdge<T>)> {
        let var = self.variable.finish();
        let init_val = constant(self.initialiser.tensor(&self.shape)?);
        let assign = ops::Assign::build(var.clone(), init_val)
                                 .validate_shape(self.validate_shape)
                                 .finish();
        Ok((var, assign))
    } 
}

impl<T: TensorType> Deref for Variable<T> {
    type Target = ops::VariableV2<T>;
    fn deref(&self) -> &Self::Target {
        &self.variable
    }
}

impl<T: TensorType> DerefMut for Variable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.variable
    }
}


pub trait Initialiser<T: TensorType> {
    fn tensor(&self, shape: &[u64]) -> Result<Tensor<T>>;
}

#[derive(Debug)]
pub struct ConstantInitialiser<T: TensorType> {
    value: T,
}

impl<T: TensorType> ConstantInitialiser<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
        }
    }
}

impl<T: TensorType> Initialiser<T> for ConstantInitialiser<T> {
    fn tensor(&self, shape: &[u64]) -> Result<Tensor<T>> {
        Tensor::from_values(shape, repeat(self.value.clone()))
    }
}


pub struct Gradients<T: TensorType> {   
    prefix: Option<String>,
    x: Vec<Box<dyn GraphEdge<T>>>,
    y: Vec<Box<dyn GraphEdge<T>>>,
    dx: Option<Vec<Box<GraphEdge<T>>>>,
    ids: Vec<usize>,
    in_graph: Cell<bool>,
}

impl<T: TensorType> Gradients<T> {
    pub fn new<I, J, U, V>(prefix: Option<&str>,
                         y: I,
                         x: J) -> Self
    where I: IntoIterator<Item=U>,
          J: IntoIterator<Item=V>,
          U: GraphEdge<T> + 'static,
          V: GraphEdge<T> + 'static, {
        let x: Vec<Box<dyn GraphEdge<T>>> = x.into_iter().map(|i| i.box_clone()).collect();
        let y = y.into_iter().map(|i| i.box_clone()).collect();
        Self {
            prefix: match prefix {
                None => None,
                Some(s) => Some(s.to_string()),
            },
            ids: (0..x.len()).map(|_| new_id()).collect(),
            x,
            y,
            dx: None,
            in_graph: Cell::new(false),
        }
    }

    pub fn dx<'a, I>(mut self, dx: I) -> Self
    where I: IntoIterator<Item=&'a (GraphEdge<T> + 'static)> {
        let dx = dx.into_iter().map(|i| i.box_clone()).collect();
        self.dx = Some(dx);
        self
    }

    /// Returns a vector of gradients in the same order as the y vector. If an Option is None
    /// it means there is no gradient and no operation so the edge cannot be used.
    pub fn edges(self, graph: &mut Graph) -> Result<Vec<Option<GradientEdge<T>>>> {
        let mut edges: Vec<Option<GradientEdge<T>>> = Vec::new();
        let rc_self = Rc::new(self);
        for i in 0..rc_self.x.len() {
            let edge = GradientEdge::<T>::new(rc_self.clone(), i);
            if edge.has_gradient(graph)? {
                edges.push(Some(edge));
            } else {
                edges.push(None);
            }
        }
        Ok(edges)
    }

    fn make_operations(&self, graph: &mut Graph) -> Result<()> {
        let xs_result: Result<Vec<Output>> = self.x.iter().map(|edge| edge.output(graph)).collect();
        let xs = xs_result?;

        let ys_result: Result<Vec<Output>> = self.y.iter().map(|edge| edge.output(graph)).collect();
        let ys = ys_result?;

        let dxs: Option<Vec<Output>> = match self.dx {
            None => None,
            Some(ref o) => {
                let dx_result: Result<Vec<Output>> = o.iter().map(|edge| edge.output(graph)).collect();
                Some(dx_result?)
            }
        };

        let outputs = graph.add_gradients(self.prefix.as_ref().map(|s| s.as_str()),
                                          &ys,
                                          &xs,
                                          dxs.as_ref().map(|v| v.as_slice()))?;
        for (output, id) in outputs.into_iter().zip(&self.ids) {
            match output {
                Some(o) => graph.record_op(*id, o.operation),
                None => {},
            }
        }
        self.in_graph.set(true);
        Ok(())
    }

    fn output(&self, graph: &mut Graph, idx: usize) -> Result<Option<Output>> {
        Ok(match self.operation(graph, idx)? {
            None => None,
            Some(op) => Some(Output{operation: op, index: 0}),
        })
    }

    fn operation(&self, graph: &mut Graph, idx: usize) -> Result<Option<Operation>> {
        if !self.in_graph.get() {
            self.make_operations(graph)?
        };

        Ok(graph.get_op_by_id(self.ids[idx]))
    }
}

#[derive(Clone)]
pub struct GradientEdge<T: TensorType> {
    parent: Rc<Gradients<T>>,
    idx: usize,
}

impl<T: TensorType> GradientEdge<T> {
    fn new(parent: Rc<Gradients<T>>, idx: usize) -> Self{
        Self {
            parent,
            idx,
        }
    }

    /// If there is no gradient between two nodes tensorflow won't create an operation
    fn has_gradient(&self, graph: &mut Graph) -> Result<bool> {
        Ok(self.parent.output(graph, self.idx)?.is_some())
    }
}

impl<T: TensorType> GraphEdge<T> for GradientEdge<T> {

    fn output(&self, graph: &mut Graph) -> Result<Output> {
        Ok(self.parent.output(graph, self.idx)?.unwrap())
    }

    fn operation(&self, graph: &mut Graph) -> Result<Operation> {
        Ok(self.parent.operation(graph, self.idx)?.unwrap())
    }

    fn box_clone(&self) -> Box<dyn GraphEdge<T>> {
        Box::new(self.clone())
    }
}

impl<T: TensorType> GraphOperation for GradientEdge<T> {
    fn tf_operation(&self, graph: &mut Graph) -> Result<Operation> {
        self.operation(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Graph, SessionOptions, Session, SessionRun};

    #[test]
    fn test_const() {
        let mut graph = Graph::new();

        let a = constant(vec![1, 2, 3]);

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        let tensor = sess.fetch(&mut graph, &a).unwrap();
        assert_eq!(tensor, vec![1, 2, 3].into());
    }

    #[test]
    fn test_add() {
        let mut graph = Graph::new();

        let a = constant(vec![1, 2, 3]);
        let b = constant(vec![2, 2, 3]);
        let c = a + b;

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        let tensor = sess.fetch(&mut graph, &c).unwrap();
        assert_eq!(tensor, vec![3, 4, 6].into());
    }

    #[test]
    fn test_sub() {
        let mut graph = Graph::new();

        let a = constant(vec![1, 2, 3]);
        let b = constant(vec![2, 2, 3]);
        let c = a - b;

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        let tensor = sess.fetch(&mut graph, &c).unwrap();
        assert_eq!(tensor, vec![-1, 0, 0].into());
    }

    #[test]
    fn test_mul() {
        let mut graph = Graph::new();

        let a = constant(vec![1, 2, 3]);
        let b = constant(vec![2, 2, 3]);
        let c = a * b;

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        let tensor = sess.fetch(&mut graph, &c).unwrap();
        assert_eq!(tensor, vec![2, 4, 9].into());
    }

    #[test]
    fn test_div() {
        let mut graph = Graph::new();

        let a = constant(vec![1, 2, 3]);
        let b = constant(vec![2, 2, 3]);
        let c = a / b;

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        let tensor = sess.fetch(&mut graph, &c).unwrap();
        assert_eq!(tensor, vec![0, 1, 1].into());
    }

    #[test]
    fn test_constant_init() {
        let mut graph = Graph::new();
        let (var, init) = Variable::<i32>::new(&[2], ConstantInitialiser::new(3)).unwrap();

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        {
            let mut run = SessionRun::new(&mut graph);
            run.add_op(&init).unwrap();
            run.run(&sess).unwrap();
        }


        let mut run = SessionRun::new(&mut graph);
        let token = run.add_edge(&var).unwrap();
        let mut result = run.run(&sess).unwrap();

        let result = result.get(token).unwrap();
        assert_eq!(result, vec![3, 3].into());
    }

    #[test]
    fn test_gradients() {
        let mut graph = Graph::new();

        let (x, init) = Variable::<f64>::new(&[1], ConstantInitialiser::new(2.1)).unwrap();
        let y = ops::Mul::new(x.clone(), constant(3.0));
        let mut grad = Gradients::new(None, vec![y.clone()], vec![x.clone()]).edges(&mut graph).unwrap();

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        {
            let mut run = SessionRun::new(&mut graph);
            run.add_op(&init).unwrap();
            run.run(&sess).unwrap();
        }


        let mut run = SessionRun::new(&mut graph);
        let token = run.add_edge(&grad[0].take().unwrap()).unwrap();
        let mut result = run.run(&sess).unwrap();

        let result = result.get(token).unwrap();
        assert_eq!(result, vec![3.0].into());
    }

    #[test]
    fn test_gradients2() {
        let mut graph = Graph::new();

        let (x, init) = Variable::<f64>::new(&[1], ConstantInitialiser::new(2.1)).unwrap();
        let y = ops::Mul::new(x.clone(), x.clone());
        let mut grad = Gradients::new(None, vec![y.clone()], vec![x.clone()]).edges(&mut graph).unwrap();

        let options = SessionOptions::new();
        let sess = Session::new(&options, &graph).unwrap();

        {
            let mut run = SessionRun::new(&mut graph);
            run.add_op(&init).unwrap();
            run.run(&sess).unwrap();
        }


        let mut run = SessionRun::new(&mut graph);
        let token = run.add_edge(&grad[0].take().unwrap()).unwrap();
        let mut result = run.run(&sess).unwrap();

        let result = result.get(token).unwrap();
        assert_eq!(result, vec![4.2].into());
    }
}
