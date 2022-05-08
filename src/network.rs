use crate::{
    commons::{der_sigmoid, error},
    hidden_layer::HiddenLayer,
    input_layer::InputLayer,
};

pub struct Network {
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    // output layer is just like a hidden layer but with less neurons
    output_layer: HiddenLayer,
}

impl Network {
    /// Create a new network with the given numbers of layers and bias.
    ///
    /// * `input`: vector of input values for the network
    /// * `hidden_num`: number of hidden layers
    /// * `hidden_size`: number of neurons in each hidden layer
    /// * `output_size`: number of neurons in the output layer
    /// * `bias`: bias value for all neurons
    pub fn new(
        input: Vec<f64>,
        hidden_num: usize,
        hidden_size: usize,
        output_size: usize,
        bias: f64,
    ) -> Network {
        let input_layer = InputLayer::new(input);

        let mut hidden_layers: Vec<HiddenLayer> = Vec::with_capacity(hidden_num);

        for i in 0..hidden_num {
            hidden_layers.push(HiddenLayer::new(hidden_size, hidden_size, bias));
        }

        let output_layer = HiddenLayer::new(output_size, hidden_size, bias);

        Network {
            input_layer: input_layer,
            hidden_layers: hidden_layers,
            output_layer: output_layer,
        }
    }

    /// Feedforeward the network with the inputs given in the constructor.
    /// Return the output of the network.
    pub fn simple_feedforward(&mut self) -> Vec<f64> {
        let mut prev_out: Vec<f64> = Vec::new();
        for (i, hidden) in self.hidden_layers.iter_mut().enumerate() {
            if i == 0 {
                // if this is the first iteration, get inputs from the input layer
                hidden.compute(self.input_layer.get_inputs());
            } else {
                hidden.compute(prev_out);
            }
            prev_out = hidden.get_output();
        }
        self.output_layer
            .compute(self.hidden_layers[self.hidden_layers.len() - 1].get_output());
        self.output_layer.get_output()
    }

    /// Train the network with the given inputs and expected outputs.
    /// ### Parameters
    /// * `expected`: expected outputs for the network
    /// * `learning_rate`: learning rate for the network
    /// * `alpha` : alpha value for the network
    pub fn train(&mut self, expected: Vec<f64>, learning_rate: f64, alpha: f64) {
        for _ in 0..100 {
            let guess = self.simple_feedforward();

            let output_error = error(&guess, &expected);

            // gradient descent vector
            let mut gradient_weights: Vec<f64> = Vec::new();
            let mut gradient_bias: Vec<f64> = Vec::new();

            // derivative of a sum of the errors with respect to the output of the neuron
            let cost_out_layer = output_error.iter().sum::<f64>() * 2.;
            print!("{},  ", cost_out_layer);

            // compute the gradient of the output layer
            // the gradient consists of:
            // 1. derivative of the cost function with respect to the weight of this neuron
            // 2. derivative of the cost function with respect to the bias of this neuron
            // repeat for all the neurons in the output layer
            for (i, neuron) in self.output_layer.get_neurons().iter_mut().enumerate() {
                for (j, input) in neuron.get_inputs().iter().enumerate() {
                    // derivate of cost function with respect to the weight of the neuron
                    //
                    // 2. * (guess[i] - output_error[i]) is a derivative of the cost function with respect to the output of this neuron
                    //
                    // der_sigmoid(neuron.single_input_weighted(j)) is a derivative of the output of this neuron (sigmoid function)
                    // with respect to the weighed j'th input of this neuron with bias
                    //
                    // input is a clean j'th input of that neuron which is a derivative of weighed j'th input of this neuron with bias
                    // with respect to the weight of this neuron

                    gradient_weights.push(
                        learning_rate
                            * 2.
                            * (cost_out_layer)
                            * der_sigmoid(neuron.single_input_weighted(j))
                            * input,
                    );

                    // derivate of cost function with respect to the bias of the neuron
                    //
                    // 2. * (guess[i] - output_error[i]) is a derivative of the cost function with respect to the output of this neuron
                    //
                    // der_sigmoid(neuron.single_input_weighted(j)) is a derivative of the output of this neuron (sigmoid function)
                    // with respect to the weighed j'th input of this neuron with bias
                    //
                    // 1 is a derivative of weighed j'th input of this neuron with bias with respect to the bias of this neuron

                    gradient_bias.push(
                        learning_rate
                            * 2.
                            * (cost_out_layer)
                            * der_sigmoid(neuron.single_input_weighted(j)),
                    );
                }
            }

            let mut change_weights: Vec<Vec<f64>> = Vec::new();
            let mut change_bias: Vec<Vec<f64>> = Vec::new();
            // change the weights and biases of the output layer
            for (i,neuron) in self.output_layer.get_neurons().iter_mut().enumerate() {
                let mut weights: Vec<f64> = neuron.get_weights().to_vec();
                let mut bias: Vec<f64> = neuron.get_biases().to_vec();
                for k in 0..change_weights.len()
                {
                    weights[k] -= alpha * (gradient_weights
                        [k + i * self.output_layer.get_neurons().len()]);
                    bias[k] -= alpha * (gradient_bias[k + i * self.output_layer.get_neurons().len()]);
                }
                change_weights.push(weights);
                change_bias.push(bias);
            }

            self.output_layer.set_weights(change_weights);
            self.output_layer.set_biases(change_bias);

            gradient_weights.clear();
            gradient_bias.clear();

            // compute the gradient of the hidden layers
            // the same as the output layer
            for hidden in self.hidden_layers.iter_mut() {
                for neuron in hidden.get_neurons().iter_mut() {
                    for (k, input) in neuron.get_inputs().iter().enumerate() {
                        gradient_weights.push(
                            learning_rate
                                * 2.
                                * (cost_out_layer)
                                * der_sigmoid(neuron.single_input_weighted(k))
                                * input,
                        );
                        gradient_bias.push(
                            learning_rate
                                * 2.
                                * (cost_out_layer)
                                * der_sigmoid(neuron.single_input_weighted(k)),
                        );
                    }
                }
            }

            let mut change_weights: Vec<Vec<f64>> = Vec::new();
            let mut change_bias: Vec<Vec<f64>> = Vec::new();
            // change the weights and biases of the hidden layers
            for (i, hidden) in self.hidden_layers.iter_mut().enumerate() {
                change_weights.clear();
                change_bias.clear();
                for (j, neuron) in hidden.get_neurons().iter_mut().enumerate() {
                    let mut weights: Vec<f64> = neuron.get_weights().to_vec();
                    let mut bias: Vec<f64> = neuron.get_biases().to_vec();
                    for k in 0..weights.len()
                    {
                        weights[k] -= alpha * (gradient_weights
                            [k + i * hidden.get_neurons().len() + j * neuron.get_biases().len()]);
                        bias[k] -= alpha * (gradient_bias[k + i * hidden.get_neurons().len() + j * neuron.get_weights().len()]);
                    }
                    change_weights.push(weights);
                    change_bias.push(bias);
                }
                hidden.set_weights(change_weights.to_vec());
                hidden.set_biases(change_bias.to_vec());
            }
        }
    }

    pub fn debug_print(&mut self) {
        println!("output neurons");
        for neuron in self.output_layer.get_neurons().iter() {
            print!("({}, {}, {})", neuron.get_weights().len(), neuron.get_biases().len(), neuron.get_inputs().len());
            println!("");
        }
        println!("hidden neurons");
        for hidden in self.hidden_layers.iter_mut() {
            for neuron in hidden.get_neurons().iter() {
                print!("({}, {}, {})", neuron.get_weights().len(), neuron.get_biases().len(), neuron.get_inputs().len());
                println!("");
            }
        }
    }
}
