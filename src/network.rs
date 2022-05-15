use crate::{
    commons::{der_sigmoid, error, sigmoid},
    hidden_layer::HiddenLayer,
    input_layer::InputLayer,
};

pub struct Network {
    pub input_layer: InputLayer,
    pub hidden_layers: Vec<HiddenLayer>,
    // output layer is just like a hidden layer.
    pub output_layer: HiddenLayer,
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

        hidden_layers.push(HiddenLayer::new(hidden_size, input_layer.values.len(), bias));

        for i in 1..hidden_num {
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
                hidden.compute(&self.input_layer.values);
            } else {
                hidden.compute(&prev_out);
            }
            prev_out = hidden.outputs.clone();
        }
        self.output_layer
            .compute(&self.hidden_layers[self.hidden_layers.len() - 1].outputs);
        self.output_layer.outputs.clone()
    }

    /// Train the network with the given inputs and expected outputs.
    /// ### Parameters
    /// * `expected`: expected outputs for the network
    /// * `learning_rate`: learning rate for the network
    /// * `alpha` : alpha value for the network
    pub fn train(&mut self, training_data: (Vec<Vec<f64>>, Vec<Vec<f64>>) , learning_rate: f64, alpha: f64) {
        let (inputs, expected) = training_data;
        // train x times
        for _ in 0..5 {
            // for each input/expected combo
            for (input, expect) in inputs.iter().zip(expected.iter()) {
                //println!("Training for input: {} with expected: {}", input[0], expect[0]);
                // train 10 times
                for _ in 0..1 {
                    self.input_layer.values = input.clone();
                    let guess = self.simple_feedforward();
                    print!("Guess for input {:?} ->  {:?}\n",&input, guess);

                    let output_error = error(&expect, &guess);
                    //println!("Error: for input {:?} -> {:?}", c, output_error);

                    // gradient descent vector
                    let mut output_weight_gradient: Vec<f64> = Vec::new();


                    // gradient of the output layer
                    // grad[i] = -2*(Ti - Oi) * Oi * (1-Oi)
                    // where Ti is the ith target output
                    // and Oi is the ith output of the output layer
                    for i in 0..self.output_layer.neurons.len() {
                        output_weight_gradient.push(
                            (-2.*(expect[i] - guess[i]))
                            * der_sigmoid(guess[i])
                        );
                    }

                    // change the weights and biases of the output layer
                    for (i,neuron) in self.output_layer.neurons.iter_mut().enumerate() {
                        for k in 0..neuron.weights.len()
                        {
                            // alpha * grad[i] * z[j,i]
                            // where z[j,i] is the sigm(wj*xj + b) of the neuron i
                            neuron.weights[k] -=
                            alpha *
                            (output_weight_gradient[i]) *
                            neuron.inputs[k];
                        }
                    }

                    let mut hidden_gradient: Vec<f64> = Vec::new();

                    // compute the gradient of the hidden layers
                    // the same as the output layer
                    // we need to reverse them, because we're going from the last to the first layer
                    let mut previous_hidden: HiddenLayer = HiddenLayer::new(1, 1, 1.);
                    for (i, hidden) in self.hidden_layers.iter_mut().rev().enumerate() {
                        if i == 0 {
                            // this is the LAST layer which means its based on the output layer gradient
                            for (j, neuron) in hidden.neurons.iter_mut().enumerate() {
                                let mut grad = 0.;
                                for (k, elem) in output_weight_gradient.iter().enumerate() {
                                    grad +=
                                    elem *
                                    self.output_layer.neurons[k].weights[j];
                                }
                                hidden_gradient.push(
                                    grad * neuron.compute()
                                );
                            }
                            // change the weights and biases of the first layers
                            for (j, neuron) in hidden.neurons.iter_mut().enumerate() {
                                for (weight, input) in neuron.weights.iter_mut().zip(neuron.inputs.iter())
                                {
                                    *weight -= alpha * hidden_gradient[j] * input;
                                }
                            }
                        } else {
                            // for the rest of the hidden layers based on the previous layer gradient
                            for (j, neuron) in hidden.neurons.iter_mut().enumerate() {
                                let mut grad = 0.;
                                for (k, elem) in hidden_gradient.iter().enumerate() {
                                    grad +=
                                    elem *
                                    previous_hidden.neurons[k].weights[j];
                                }
                                hidden_gradient[j] = grad * neuron.compute();

                            }
                            // change the weights and biases of the first layers
                            for (j, neuron) in hidden.neurons.iter_mut().enumerate() {
                                for (weight, input) in neuron.weights.iter_mut().zip(neuron.inputs.iter())
                                {
                                    *weight -= alpha * hidden_gradient[j] * input;
                                }
                            }
                        }
                        previous_hidden = hidden.clone();
                    }
                }
                //momentum -= momentum * 0.95;
                //println!("training result: {:?}", self.simple_feedforward());
            }
        }
    }

    pub fn batch_train(&mut self, training_data: (Vec<Vec<f64>>, Vec<Vec<f64>>) , learning_rate: f64, alpha: f64) {
        todo!()
    }

    pub fn debug_print(&mut self) {
        println!("output neurons");
        for neuron in self.output_layer.neurons.iter() {
            print!("({}, {}, {})", neuron.weights.len(), neuron.bias.len(), neuron.inputs.len());
            println!("");
        }
        println!("hidden neurons");
        for hidden in self.hidden_layers.iter_mut() {
            for neuron in hidden.neurons.iter() {
                print!("({}, {}, {})", neuron.weights.len(), neuron.bias.len(), neuron.inputs.len());
                println!("");
            }
        }
    }
}
