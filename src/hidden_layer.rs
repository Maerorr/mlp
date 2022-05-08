use crate::neuron::{Neuron};
use crate::commons::{gen_weights};

pub struct HiddenLayer {
    neurons: Vec<Neuron>,
    outputs: Vec<f64>,
}

impl HiddenLayer {

    /// Creates a new hidden layer with the given number of neurons
    /// and the neurons each have given number of inputs.
    pub fn new(hidden_neurons: usize, connections: usize ,bias: f64) -> HiddenLayer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(hidden_neurons);
        for _ in 0..hidden_neurons {
            // Generate random weights for each neuron and add the bias
            neurons.push(Neuron::new(gen_weights(connections), bias));
        }
        HiddenLayer {
            neurons: neurons,
            outputs: Vec::new(),
        }
    }

    /// Get inputs from the previous layer and
    /// compute outputs of all the neurons
    pub fn compute(&mut self, inputs: Vec<f64>) {
        self.outputs.clear();
        // Set the inputs of each neuron
        for neuron in self.neurons.iter_mut() {
            neuron.set_inputs(inputs.clone());
        }
        // compute outputs of all the neurons
        for neuron in self.neurons.iter_mut() {
            self.outputs.push(neuron.compute());
        }
    }

    pub fn get_output(&self) -> Vec<f64> {
        self.outputs.clone()
    }

    pub fn get_neurons(&self) -> Vec<Neuron> {
        self.neurons.clone()
    }

    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.set_weights(weights[i].clone());
        }
    }

    pub fn set_biases(&mut self, biases: Vec<Vec<f64>>) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.set_biases(biases[i].clone());
        }
    }
}
