use super::{Gradients, TensorType, GraphRefEdge, GraphEdge, Graph, Result, Variable, constant, ConstantInitialiser};
use super::ops::{ApplyGradientDescent, NoOp, con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_QINT8_or_DT_QUINT8_or_DT_QINT32_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF_or_DT_UINT32_or_DT_UINT64, Sub, Mul};

/// Returns an operation that performs gradient descent on vars, based on cost and alpha
/// prefix is prefixed to the names of the gradient operations
pub fn gradient_descent<T, RefT, RefIter, AlphaT, CostT>(graph: &mut Graph,
                                                     vars: RefIter,
                                                     alpha: AlphaT,
                                                     cost: CostT,
                                                     prefix: Option<&str>) -> Result<NoOp>
where
  T: con_or_DT_FLOAT_or_DT_DOUBLE_or_DT_INT32_or_DT_UINT8_or_DT_INT16_or_DT_INT8_or_DT_COMPLEX64_or_DT_INT64_or_DT_QINT8_or_DT_QUINT8_or_DT_QINT32_or_DT_BFLOAT16_or_DT_UINT16_or_DT_COMPLEX128_or_DT_HALF_or_DT_UINT32_or_DT_UINT64 + TensorType,
  RefT: GraphRefEdge<T> + Clone + 'static,
  RefIter: IntoIterator<Item=RefT> + Clone + 'static,
  AlphaT: GraphEdge<T> + Clone + 'static,
  CostT: GraphEdge<T> + 'static, {
    let gradients = Gradients::new(prefix, Some(cost), vars.clone()).edges(graph)?;

    let mut op = NoOp::build();

    for (gradient, var) in gradients.into_iter().zip(vars) {
      if let Some(gradient) = gradient {
        op.control_input(ApplyGradientDescent::new(var, alpha.clone(), gradient));
      }
    }

    Ok(op)
}


#[cfg(test)]
mod tests {
  use super::*;
  use super::super::{Graph, SessionOptions, Session, SessionRun, Variable};

  #[test]
  fn test_gradient_descent() {
    let mut graph = Graph::new();

    let (var, init) = Variable::<f64>::new(&[1], ConstantInitialiser::new(4.0)).unwrap();
    let diff = Sub::new(var.clone(), constant(2.0));
    let cost = Mul::new(diff.clone(), diff.clone());

    let train = gradient_descent(&mut graph, Some(var.clone()), constant(0.1), cost, None).unwrap();

    let options = SessionOptions::new();
    let sess = Session::new(&options, &graph).unwrap();

    // Initialise variables
    {
      let mut run = SessionRun::new(&mut graph);
      run.add_op(&init).unwrap();
      run.run(&sess).unwrap();
    }

    // Run one step of training
    {
      let mut run = SessionRun::new(&mut graph);
      run.add_op(&train).unwrap();
      run.run(&sess).unwrap();
    }

    let result = sess.fetch(&mut graph, &var).unwrap();
    assert_eq!(result[0], 3.6);
  }
}