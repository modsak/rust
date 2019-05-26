use super::*;

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_const() {
    println!("1");
    let mut graph = Graph::new();
    println!("2");
    let a = ops::Const::<f32>::new(Box::new(Tensor::<f32>::new(&vec![1,2]))).finish();
    println!("3");
    let options = SessionOptions::new();
    println!("4");
    let sess = Session::new(&options, &graph).unwrap();
    println!("5");
    let result = sess.fetch(&mut graph, &a.output()).unwrap();
    // println!("6");
    // assert_eq!(result, 2_f32.into());
    // println!("7");
  }
}