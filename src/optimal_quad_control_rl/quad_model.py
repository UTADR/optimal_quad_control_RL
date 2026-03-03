import numba
import numpy as np
from numpy.typing import NDArray

from optimal_quad_control_rl import rotation


def f_func(state: NDArray, control: NDArray, params: NDArray) -> NDArray:
    """
    JAX explicit port of the symbolic quadcopter equations of motion.
    Expects state, control, and params to be unpacked along the first dimension.
    """
    # 1. Unpack NDArrays
    num_batch = state.shape[0]
    velocity = state[..., 3:6]
    angles = state[..., 6:9]
    body_rates = state[..., 9:12]
    w = state[..., 12:16]
    u = control

    (
        k_x,
        k_y,
        k_w,
        k_p1,
        k_p2,
        k_p3,
        k_p4,
        k_q1,
        k_q2,
        k_q3,
        k_q4,
        k_r1,
        k_r2,
        k_r3,
        k_r4,
        k_r5,
        k_r6,
        k_r7,
        k_r8,
        tau,
        k,
        w_min,
        w_max,
    ) = params.T

    g = 9.81
    w_min_n = 0.0
    w_max_n = 3000.0

    res = np.empty((num_batch, 16))
    # Declare views/aliases to slices of res; we will reuse these as variables for other computation
    angular_accel = res[..., 9:12]
    d_w = res[..., 12:16]

    # 4. Body Velocity (R^T @ v)
    vb = rotation.euler_angles_rotate_point(angles, velocity)
    # 5. Motor RPMs (rad/s)
    W = (w + 1.0) / 2.0 * (w_max_n - w_min_n) + w_min_n

    # Motor commands scaled to [0,1]
    U = (u + 1.0) / 2.0

    # Steady-state rpm target
    Wc = (w_max - w_min)[..., None] * np.sqrt(
        k[..., None] * U**2 + (1.0 - k[..., None]) * U
    ) + w_min[..., None]

    # First-order delay derivative (rad/s)
    d_W = (Wc - W) / tau[..., None]

    # Normalized motor speeds derivative
    d_w[:] = d_W / (w_max_n - w_min_n) * 2.0

    # 6. Aerodynamics (Thrust and Drag)
    W_sum = W.sum(axis=-1)
    W_sq = W**2

    force = np.empty((num_batch, 3))
    force[..., 0] = -k_x * vb[..., 0] * W_sum
    force[..., 1] = -k_y * vb[..., 1] * W_sum
    force[..., 2] = -k_w * W_sq.sum(axis=-1)

    # Moments

    angular_accel[..., 0] = (
        -k_p1 * W_sq[..., 0]
        - k_p2 * W_sq[..., 1]
        + k_p3 * W_sq[..., 2]
        + k_p4 * W_sq[..., 3]
    )
    angular_accel[..., 1] = (
        -k_q1 * W_sq[..., 0]
        + k_q2 * W_sq[..., 1]
        - k_q3 * W_sq[..., 2]
        + k_q4 * W_sq[..., 3]
    )
    angular_accel[..., 2] = (
        -k_r1 * W[..., 0]
        + k_r2 * W[..., 1]
        + k_r3 * W[..., 2]
        - k_r4 * W[..., 3]
        - k_r5 * d_W[..., 0]
        + k_r6 * d_W[..., 1]
        + k_r7 * d_W[..., 2]
        - k_r8 * d_W[..., 3]
    )

    res[..., 0:3] = velocity
    res[..., 3:6] = rotation.euler_angles_rotate_point_inv(angles, force)
    res[..., 5] += g
    res[..., 6:9] = rotation.euler_angles_kinematic_rates(angles, body_rates)
    return res


def normalize(x, x_min, x_max):
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


def param_encoding(params: NDArray) -> NDArray:
    """
    JAX explicit port of the symbolic parameter encoding for a single environment.
    """
    (
        k_x,
        k_y,
        k_w,
        k_p1,
        k_p2,
        k_p3,
        k_p4,
        k_q1,
        k_q2,
        k_q3,
        k_q4,
        k_r1,
        k_r2,
        k_r3,
        k_r4,
        k_r5,
        k_r6,
        k_r7,
        k_r8,
        tau,
        k,
        w_min,
        w_max,
    ) = params.T

    # De-normalize thrust and moment constants by scaling with w_max
    k_wn = k_w * (w_max**2)
    k_pn = (k_p1 + k_p2 + k_p3 + k_p4) / 4.0 * (w_max**2)
    k_qn = (k_q1 + k_q2 + k_q3 + k_q4) / 4.0 * (w_max**2)
    k_rn = (k_r1 + k_r2 + k_r3 + k_r4) / 4.0 * w_max
    k_rdn = (k_r5 + k_r6 + k_r7 + k_r8) / 4.0 * w_max

    res = np.empty((len(params), 9))

    res[..., 0] = normalize(k_wn, 1.0e01, 3.0e01)
    res[..., 1] = normalize(k_pn, 2.0e02, 8.0e02)
    res[..., 2] = normalize(k_qn, 2.0e02, 8.0e02)
    res[..., 3] = normalize(k_rn, 2.0e01, 8.0e01)
    res[..., 4] = normalize(k_rdn, 2.0e00, 8.0e00)
    res[..., 5] = normalize(k, 0.0, 1.0)
    res[..., 6] = normalize(tau, 0.01, 0.1)
    res[..., 7] = normalize(w_min, 0.0, 500.0)
    res[..., 8] = normalize(w_max, 3000.0, 5000.0)
    return res
