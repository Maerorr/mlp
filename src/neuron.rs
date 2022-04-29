use crate::commons::sigmoid;

pub struct Neuron {
    inputs: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    /// NOTE: last element of weights is the bias weight
    pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
        Neuron {
            inputs: Vec::new(),
            weights: weights,
            bias: bias,
        }
    }

    pub fn set_inputs(&mut self, inputs: Vec<f64>) {
        self.inputs = inputs;
    }

    /// Calculate the output of the neuron
    /// by summing all the inputs multiplied by the weights
    pub fn compute(&mut self) -> f64 {
        let mut sum = 0.;
        //println!("{:?}", self.inputs.len());
        for iter in self.inputs.iter().zip(self.weights.iter()) {
            let (input, weight) = iter;
            sum += input * weight;
        }
        // Calculate the output using the sigmoid function and add the bias
        sigmoid(sum + self.bias*self.weights[self.weights.len()-1])
    }
}