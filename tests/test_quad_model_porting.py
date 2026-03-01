import numpy as np
import pytest
from pytest import fixture
from sympy import Array, Matrix, cos, lambdify, sin, sqrt, symbols, tan

from optimal_quad_control_rl import quad_model
from optimal_quad_control_rl.randomization import randomization_3inch_10_percent


@fixture
def num_envs():
    return 10


@fixture
def rng():
    return np.random.default_rng(seed=42)


@fixture
def test_inputs(rng: np.random.Generator, num_envs):
    # 1. Generate Controls (Actions)
    # The action space expects normalized motor commands between [-1, 1]
    controls = rng.uniform(size=(num_envs, 4), low=-1.0, high=1.0)

    # 2. Generate States
    # Mimicking the exact initialization bounds from Quadcopter3DGates.reset_()
    pos = rng.uniform(size=(num_envs, 3), low=-2.0, high=2.0)  # x, y, z
    vel = rng.uniform(size=(num_envs, 3), low=-0.5, high=0.5)  # vx, vy, vz
    phi_theta = rng.uniform(
        size=(num_envs, 2),
        low=-np.pi / 9,
        high=np.pi / 9,
    )  # phi, theta
    psi = rng.uniform(size=(num_envs, 1), low=-np.pi, high=np.pi)  # psi
    rates = rng.uniform(size=(num_envs, 3), low=-0.1, high=0.1)  # p, q, r
    rpms = rng.uniform(size=(num_envs, 4), low=-1.0, high=1.0)  # w1, w2, w3, w4

    # Stack them vertically to get the (16, num_envs) shape
    states = np.hstack([pos, vel, phi_theta, psi, rates, rpms])

    return states, controls


@fixture
def params():
    return symbols(
        "k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max"
    )


@fixture
def symbolic_f_func(params):
    state = symbols("x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4")
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = state
    control = symbols("U_1 U_2 U_3 U_4")  # normalized motor commands between [-1,1]
    u1, u2, u3, u4 = control

    g = 9.81
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
    ) = params

    # Rotation matrix
    Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    # Body velocity
    vbx, vby, vbz = R.T @ Matrix([vx, vy, vz])

    # normalized motor speeds to rad/s
    w_min_n = 0.0
    w_max_n = 3000.0
    W1 = (w1 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
    W2 = (w2 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
    W3 = (w3 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
    W4 = (w4 + 1) / 2 * (w_max_n - w_min_n) + w_min_n

    # motor commands scaled to [0,1]
    U1 = (u1 + 1) / 2
    U2 = (u2 + 1) / 2
    U3 = (u3 + 1) / 2
    U4 = (u4 + 1) / 2

    # first order delay:
    # the steadystate rpm motor response to the motor command U is described by:
    # Wc = (w_max-w_min)*sqrt(k U**2 + (1-k)*U) + w_min
    Wc1 = (w_max - w_min) * sqrt(k * U1**2 + (1 - k) * U1) + w_min
    Wc2 = (w_max - w_min) * sqrt(k * U2**2 + (1 - k) * U2) + w_min
    Wc3 = (w_max - w_min) * sqrt(k * U3**2 + (1 - k) * U3) + w_min
    Wc4 = (w_max - w_min) * sqrt(k * U4**2 + (1 - k) * U4) + w_min

    # rad/s
    d_W1 = (Wc1 - W1) / tau
    d_W2 = (Wc2 - W2) / tau
    d_W3 = (Wc3 - W3) / tau
    d_W4 = (Wc4 - W4) / tau

    # normalized motor speeds d/dt[W - w_min_n)/(w_max_n-w_min_n)*2 - 1]
    d_w1 = d_W1 / (w_max_n - w_min_n) * 2
    d_w2 = d_W2 / (w_max_n - w_min_n) * 2
    d_w3 = d_W3 / (w_max_n - w_min_n) * 2
    d_w4 = d_W4 / (w_max_n - w_min_n) * 2

    # Thrust and Drag
    T = -k_w * (W1**2 + W2**2 + W3**2 + W4**2)
    Dx = -k_x * vbx * (W1 + W2 + W3 + W4)
    Dy = -k_y * vby * (W1 + W2 + W3 + W4)

    # Moments
    Mx = -k_p1 * W1**2 - k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2
    My = -k_q1 * W1**2 + k_q2 * W2**2 - k_q3 * W3**2 + k_q4 * W4**2
    Mz = (
        -k_r1 * W1
        + k_r2 * W2
        + k_r3 * W3
        - k_r4 * W4
        - k_r5 * d_W1
        + k_r6 * d_W2
        + k_r7 * d_W3
        - k_r8 * d_W4
    )

    # Dynamics
    d_x = vx
    d_y = vy
    d_z = vz

    d_vx, d_vy, d_vz = Matrix([0, 0, g]) + R @ Matrix([Dx, Dy, T])

    d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
    d_theta = q * cos(phi) - r * sin(phi)
    d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

    d_p = Mx
    d_q = My
    d_r = Mz

    # State space model
    f = [
        d_x,
        d_y,
        d_z,
        d_vx,
        d_vy,
        d_vz,
        d_phi,
        d_theta,
        d_psi,
        d_p,
        d_q,
        d_r,
        d_w1,
        d_w2,
        d_w3,
        d_w4,
    ]

    # lambdify
    return lambdify((Array(state), Array(control), Array(params)), Array(f), "numpy")


@fixture
def symbolic_param_encoding(params):
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
    ) = params
    # PARAMETER ENCODING (used for parameter input)
    # normalize thrust and moment constants by scaling with w_max
    k_wn = k_w * (w_max**2)
    k_pn = (k_p1 + k_p2 + k_p3 + k_p4) / 4 * (w_max**2)
    k_qn = (k_q1 + k_q2 + k_q3 + k_q4) / 4 * (w_max**2)
    k_rn = (k_r1 + k_r2 + k_r3 + k_r4) / 4 * (w_max)
    k_rdn = (k_r5 + k_r6 + k_r7 + k_r8) / 4 * (w_max)

    # normalize to [-1,1] based on min and max expected values
    normalize = lambda x, x_min, x_max: 2 * (x - x_min) / (x_max - x_min) - 1
    k_w_encoding = normalize(k_wn, 1.0e01, 3.0e01)
    k_p_encoding = normalize(k_pn, 2.0e02, 8.0e02)
    k_q_encoding = normalize(k_qn, 2.0e02, 8.0e02)
    k_r_encoding = normalize(k_rn, 2.0e01, 8.0e01)
    k_rd_encoding = normalize(k_rdn, 2.0e00, 8.0e00)
    k_encoding = normalize(k, 0.0, 1.0)
    tau_encoding = normalize(tau, 0.01, 0.1)
    w_min_encoding = normalize(w_min, 0, 500)
    w_max_encoding = normalize(w_max, 3000, 5000)

    # lambdify
    param_encoding = lambdify(
        (Array(params),),
        Array(
            [
                k_w_encoding,
                k_p_encoding,
                k_q_encoding,
                k_r_encoding,
                k_rd_encoding,
                k_encoding,
                tau_encoding,
                w_min_encoding,
                w_max_encoding,
            ]
        ),
        "numpy",
    )
    return param_encoding


@pytest.mark.parametrize("num_envs", [10, 64, 100])
def test_f_func_porting(num_envs, params, symbolic_f_func, test_inputs):
    x, u = test_inputs
    p = randomization_3inch_10_percent(num_envs)

    param_dict = p
    # 'res' shape is (10, 23)
    p = np.column_stack([param_dict[p.name] for p in params])

    # --- Test 1: Dynamics (f_func) ---
    result_dyn = quad_model.f_func(x, u, p)
    expected_dyn = symbolic_f_func(x.T, u.T, p.T).T
    np.testing.assert_allclose(expected_dyn, result_dyn)


@pytest.mark.parametrize("num_envs", [10, 64, 100])
def test_param_encoding_porting(num_envs, params, symbolic_param_encoding):
    res = randomization_3inch_10_percent(num_envs)
    param_dict = res
    res = np.column_stack([param_dict[p.name] for p in params])

    # --- Test 2: Parameter Encoding ---
    # analytical takes (10, 23), vmaps out to (9, 10)
    result_enc = quad_model.param_encoding(res)

    # symbolic lambdify takes (23, 10), computes (9, 10)
    expected_enc = symbolic_param_encoding(res.T).T

    np.testing.assert_allclose(expected_enc, result_enc)
