pub struct neuron {
    inputs: Vec<f64>,
    weights: Vec<f64>,
    output: f64,
}

impl neuron {
    pub fn new(inputs: Vec<f64>, weights: Vec<f64>) -> neuron {
        neuron {
            inputs: inputs,
            weights: weights,
            output: 0.,
        }
    }

    pub fn compute(&mut self) {
        let mut sum = 0.;
        for i in 0..self.inputs.len() {
            sum += self.inputs[i] * self.weights[i];
        }
        self.output = sigmoid(sum);
    }
}