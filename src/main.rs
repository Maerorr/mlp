use crate::network::Network;

mod commons;
mod neuron;
mod hidden_layer;
mod input_layer;
mod network;

#[allow(dead_code)]
struct Parameters {
    in_neurons: usize,
    hidden_neurons: usize,
    out_neurons: usize,
}

fn main() {
    let mut network = Network::new(vec![1.], 5, 5, 1, 0.5);
    network.simple_feedforward();
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut expected: Vec<Vec<f64>> = Vec::new();

    for i in 1..10 {
        data.push(vec![i as f64]);
        expected.push(vec![1./i as f64]);
    }

    network.train((data, expected), 0.6, 0.9);

    //network.debug_print();
    // for (data, expected) in data.chunks_exact(10).zip(expected.chunks_exact(10)) {
    //     network.batch_train((data.to_vec(), expected.to_vec()), 0.6, 0.9);
    // }


    for i in 1..10 {
        network.input_layer.values = vec![i as f64];

        println!("{}: {:?}", i , network.simple_feedforward());
    }

    //println!("{:?}", network.simple_feedforward());
    //network.debug_print();
}
