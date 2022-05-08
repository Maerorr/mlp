use crate::network::Network;

mod commons;
mod neuron;
mod hidden_layer;
mod input_layer;
mod network;

struct Parameters {
    in_neurons: usize,
    hidden_neurons: usize,
    out_neurons: usize,
}

fn main() {
    let mut network = Network::new(vec![1., 10., 2., 5.], 10, 5, 2, 0.5);
    print!("{:?}", network.simple_feedforward());
    let expected: Vec<f64> = vec![1., 2.];
    network.train(expected.to_vec(), 0.9, 0.9);

    println!("{:?}", network.simple_feedforward());
    //network.debug_print();
}
