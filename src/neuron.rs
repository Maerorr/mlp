use std::vec;

use crate::commons::sigmoid;

pub struct Neuron {
    pub inputs: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
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
