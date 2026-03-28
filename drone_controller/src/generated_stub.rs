// STUB — replaced by export.py once a trained model is available.
// Architecture: Linear(20→64, ReLU) → Linear(64→64, ReLU) → Linear(64→4)
// Weights are zeroed; constants are representative defaults.

pub const NN_INPUT_SIZE: usize = 20;
#[allow(dead_code)]
pub const HIDDEN_SIZE: usize = 64;
pub const NN_OUTPUT_SIZE: usize = 4;
pub const NUM_GATES: usize = 8;
pub const GATES_AHEAD: usize = 1;
pub const W_MIN: f32 = 1000.0;
pub const W_MAX: f32 = 4000.0;
pub const U_MAX: f32 = 1.0;

// Weights stored column-major (out_features × in_features, Fortran order).
pub const WEIGHTS_FC1_DATA: [f32; 1280] = [0.0; 1280]; // 64 × 20
pub const BIASES_FC1_DATA: [f32; 64] = [0.0; 64];
pub const WEIGHTS_FC2_DATA: [f32; 4096] = [0.0; 4096]; // 64 × 64
pub const BIASES_FC2_DATA: [f32; 64] = [0.0; 64];
pub const WEIGHTS_FC3_DATA: [f32; 256] = [0.0; 256]; // 4 × 64
pub const BIASES_FC3_DATA: [f32; 4] = [0.0; 4];

pub const OUTPUT_STD: [f32; NN_OUTPUT_SIZE] = [1.0; NN_OUTPUT_SIZE];

pub const GATE_POS: [[f32; 3]; NUM_GATES] = [[0.0; 3]; NUM_GATES];
pub const GATE_YAW: [f32; NUM_GATES] = [0.0; NUM_GATES];
pub const GATE_POS_REL: [[f32; 3]; NUM_GATES] = [[0.0; 3]; NUM_GATES];
pub const GATE_YAW_REL: [f32; NUM_GATES] = [0.0; NUM_GATES];
#[allow(dead_code)]
pub const START_POS: [f32; 3] = [0.0; 3];
#[allow(dead_code)]
pub const START_YAW: f32 = 0.0;

use crate::nn::{linear, relu};
use nalgebra::{SMatrixView, SVector};

pub fn nn_forward(x: &nalgebra::SVector<f32, NN_INPUT_SIZE>) -> nalgebra::SVector<f32, NN_OUTPUT_SIZE> {
    let w1 = SMatrixView::<f32, HIDDEN_SIZE, NN_INPUT_SIZE>::from_slice(&WEIGHTS_FC1_DATA);
    let b1 = SVector::<f32, HIDDEN_SIZE>::from_column_slice(&BIASES_FC1_DATA);
    let h1 = relu(&linear(&w1, &b1, x));
    let w2 = SMatrixView::<f32, HIDDEN_SIZE, HIDDEN_SIZE>::from_slice(&WEIGHTS_FC2_DATA);
    let b2 = SVector::<f32, HIDDEN_SIZE>::from_column_slice(&BIASES_FC2_DATA);
    let h2 = relu(&linear(&w2, &b2, &h1));
    let w3 = SMatrixView::<f32, NN_OUTPUT_SIZE, HIDDEN_SIZE>::from_slice(&WEIGHTS_FC3_DATA);
    let b3 = SVector::<f32, NN_OUTPUT_SIZE>::from_column_slice(&BIASES_FC3_DATA);
    let out = linear(&w3, &b3, &h2);
    out
}
