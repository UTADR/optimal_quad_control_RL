use drone_controller::{ADRController, OUTPUT_STD};
use pyo3::prelude::*;

/// Python-visible wrapper around the gate-traversal RL controller.
#[pyclass]
struct NNController {
    inner: ADRController,
}

#[pymethods]
impl NNController {
    #[new]
    fn new() -> Self {
        Self {
            inner: ADRController::new(),
        }
    }

    /// Reset the gate index to 0 (call before each episode).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Run one control step.
    ///
    /// Args:
    ///     world_state: 16 floats — [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r, w1-w4]
    ///                  where w1-w4 are physical RPMs.
    ///     deterministic: suppress stochastic action noise (default True).
    ///
    /// Returns:
    ///     List of 4 normalised motor commands in [0, 1].
    #[pyo3(signature = (world_state, deterministic=true))]
    fn control(&mut self, world_state: Vec<f32>, deterministic: bool) -> PyResult<Vec<f32>> {
        let ws: [f32; 16] = world_state.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("world_state must have exactly 16 elements")
        })?;
        let cmds = if deterministic {
            self.inner.control(&ws.into(), |_| {})
        } else {
            self.inner.control(&ws.into(), |out| {
                for (o, &std) in out.iter_mut().zip(OUTPUT_STD.iter()) {
                    let u1: f32 = rand::random();
                    let u2: f32 = rand::random();
                    *o += std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                }
            })
        };
        Ok(cmds.data.as_slice().to_vec())
    }
}

#[pymodule]
fn nn_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NNController>()?;
    Ok(())
}
