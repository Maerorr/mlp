use std::f64::consts::E;

use rand::{self, Rng};

/// Generate a random weight between 0 and 1.
pub fn gen_weights(size: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights: Vec<f64> = Vec::with_capacity(size);
     for _ in 0..size {
        weights.push(rng.gen::<f64>() * 2. - 1.); // scale to <-1, 1>
    }
    weights
}

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-x))
}

pub fn der_sigmoid(x: f64) -> f64 {
    // E.powf(-x)/(1. + E.powf(-x)).powf(2.)
    // this is also equal to:
    sigmoid(x) * (1. - sigmoid(x))
}

pub fn error(guess: &Vec<f64>, expected: &Vec<f64>) -> Vec<f64> {
    let mut error: Vec<f64> = Vec::with_capacity(guess.len());
    for (g, e) in guess.iter().zip(expected.iter()) {
        error.push(g - e);
    }
    error
}
