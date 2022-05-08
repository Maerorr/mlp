use std::vec;

use crate::commons::sigmoid;

pub struct Neuron {
    inputs: Vec<f64>,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl Neuron {
    /// Create a new neuron with the given weights and bias
    ///
    /// ~NOTE: last element of weights is the bias weight~
    pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
        Neuron {
            inputs: Vec::new(),
            weights: weights.to_vec(),
            bias: vec![bias; weights.len()],
        }
    }

    pub fn get_inputs(&self) -> Vec<f64> {
        self.inputs.clone()
    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn get_biases(&self) -> Vec<f64> {
        self.bias.clone()
    }

    pub fn set_inputs(&mut self, inputs: Vec<f64>) {
        self.inputs = inputs;
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn set_biases(&mut self, biases: Vec<f64>) {
        self.bias = biases;
    }

    /// Calculate the output of the neuron
    /// by summing all the inputs multiplied by the weights
    pub fn compute(&mut self) -> f64 {
        let mut sum = 0.;

        for (i, iter) in self.inputs.iter().zip(self.weights.iter()).enumerate() {
            let (input, weight) = iter;
            sum += input * weight;
            sum += self.bias[i];
        }
        // Calculate the output using the sigmoid function and add the bias
        sigmoid(sum)
    }

    /// Returns a single input of a neuron
    /// multiplied by weight and added to the bias.
    /// _For backpropagation purposes._
    pub fn single_input_weighted(&mut self, n: usize) -> f64 {
        self.inputs[n] * self.weights[n] + self.bias[n]
    }
}

impl Clone for Neuron {
    fn clone(&self) -> Neuron {
        Neuron {
            inputs: self.inputs.clone(),
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}
