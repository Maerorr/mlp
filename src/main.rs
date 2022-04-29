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
    let mut network = Network::new(vec![1., 1., 0., 0.], 5, 1, 0.5);
    let output = network.simple_feedforward();
    println!("{:?}", output);
}
