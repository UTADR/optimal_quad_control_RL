use nalgebra::{Const, Matrix, SVector, Storage, Vector};

/// Applies the ReLU activation function to each element of the input vector.
pub fn relu<const N: usize>(input: &SVector<f32, N>) -> SVector<f32, N> {
    input.map(|x| x.max(0.0))
}

/// Applies the hyperbolic tangent activation function to each element of the input vector.
pub fn tanh<const N: usize>(input: &SVector<f32, N>) -> SVector<f32, N> {
    input.map(libm::tanhf)
}

/// Applies a linear transformation to the input (dim: N) using the provided weights (dim: P x N) and biases (dim: P).
///
/// # Arguments
/// - `weights`: A matrix of shape (P, N) containing the weights for the linear transformation. This
///   is generic over storage types to accommodate viewing over flat arrays or using owned matrices.
/// - `biases`: A vector of shape (P) containing the biases for the linear transformation. This is
///   also generic over storage types.
/// - `input`: A vector of shape (N) containing the input data. This is not generic because the
///   controller driver will own the input data.
///
/// # Returns
/// A vector of shape (P) resulting from the linear transformation of the input.
pub fn linear<const N: usize, const P: usize, SW, SB>(
    weights: &Matrix<f32, Const<P>, Const<N>, SW>,
    biases: &Vector<f32, Const<P>, SB>,
    input: &SVector<f32, N>,
) -> SVector<f32, P>
where
    SW: Storage<f32, Const<P>, Const<N>>,
    SB: Storage<f32, Const<P>>,
{
    weights * input + biases
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrixView;

    // Check that linear can map a constant, flat, weight array into a smatrix and pass to linear
    #[test]
    fn test_linear_flat_weights_mapping() {
        const WEIGHTS: [f32; 60] = [1.0; 60]; // 12x5 matrix flattened
        let weights = SMatrixView::<f32, 12, 5>::from_slice(&WEIGHTS);
        let biases = SVector::<f32, 12>::from_element(5.0);
        let input = SVector::<f32, 5>::from_element(2.0);

        let output = linear(&weights, &biases, &input);
        // Each output element should be 1.0*2.0*5 + 5.0 = 15.0
        assert_eq!(output, SVector::<f32, 12>::from_element(15.0));
    }
}
