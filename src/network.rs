use crate::{input_layer::InputLayer, hidden_layer::{HiddenLayer}};

pub struct Network {
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    // output layer is just like a hidden layer but with less neurons
    output_layer: HiddenLayer,
}

impl Network {
    /// Create a new network with the given numbers of layers and bias.
    pub fn new(input: Vec<f64>, hidden_size: usize, output_size: usize, bias: f64) -> Network {
        let input_layer = InputLayer::new(input);
        let mut hidden_layers: Vec<HiddenLayer> = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            hidden_layers.push(HiddenLayer::new(hidden_size, bias));
        }
        let output_layer = HiddenLayer::new(output_size, bias);
        Network {
            input_layer: input_layer,
            hidden_layers: hidden_layers,
            output_layer: output_layer,
        }
    }

    pub fn simple_feedforward(&mut self) -> Vec<f64> {
        for (i, hidden) in self.hidden_layers.iter_mut().enumerate() {
            // temp is the previous layer's outputs
            let mut temp: Vec<f64> = Vec::new();
            if i == 0 {
                hidden.compute(self.input_layer.get_inputs());
                temp = hidden.get_output();
            }
            hidden.compute(temp);
            temp = hidden.get_output();
        }
        println!("here");
        self.output_layer.compute(self.hidden_layers[self.hidden_layers.len()-1].get_output());
        self.output_layer.get_output()
    }
}