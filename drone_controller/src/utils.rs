use nalgebra::SVector;

pub struct Box<const N: usize> {
    lb: f32,
    ub: f32,
}

impl<const N: usize> Box<N> {
    pub const fn new(lb: f32, ub: f32) -> Self {
        Self { lb, ub }
    }

    pub fn clamp_unproject(&self, nn_output: &SVector<f32, N>) -> SVector<f32, N> {
        (nn_output.map(|x| x.clamp(self.lb, self.ub)).add_scalar(1.0)) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;

    #[test]
    fn test_clamp_unproject() {
        let b = Box::<3>::new(-1.0, 1.0);
        let nn_output = SVector::<f32, 3>::new(-2.0, 0.0, 2.0);
        let result = b.clamp_unproject(&nn_output);
        assert_eq!(result, SVector::<f32, 3>::new(0.0, 0.5, 1.0));
    }
}
