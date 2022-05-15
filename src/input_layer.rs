
pub struct InputLayer {
    pub values: Vec<f64>,
}

impl InputLayer {
    pub fn new(values: Vec<f64>) -> InputLayer {
        InputLayer {
            values: values,
        }
    }
}
