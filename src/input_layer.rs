
pub struct InputLayer {
    values: Vec<f64>,
}

impl InputLayer {
    pub fn new(values: Vec<f64>) -> InputLayer {
        InputLayer {
            values: values,
        }
    }
    pub fn get_inputs(&self) -> Vec<f64> {
        self.values.to_vec()
    }
}
