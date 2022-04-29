use crate::neuron::{Neuron, self};
use crate::commons::{gen_weights};

pub struct HiddenLayer {
    neurons: Vec<Neuron>,
    outputs: Vec<f64>,
}

impl HiddenLayer {
    pub fn new(hidden_neurons: usize, bias: f64) -> HiddenLayer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(hidden_neurons);
        for _ in 0..hidden_neurons {
            // Generate random weights for each neuron and add the bias
            neurons.push(Neuron::new(gen_weights(hidden_neurons+1), bias));
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
}