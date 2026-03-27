import ctypes
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from optimal_quad_control_rl.export import (
    NNCtx,
    build_library,
    build_rust,
    emit_controller,
    generate_neural_network,
    generate_rust,
)
from optimal_quad_control_rl.quad_race_env import (
    Quadcopter3DGates,
    gate_pos,
    gate_yaw,
    start_pos,
)
from optimal_quad_control_rl.randomization import (
    params_5inch,
    randomization_fixed_params_5inch,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# Fixed inputs — seeded, never change
_rng = np.random.default_rng(42)
NN_FORWARD_INPUTS = _rng.random((1000, 20)).astype(np.float32)
# world_state: indices 12-15 must be in physical RPM range for nn_control
_ws = _rng.random((500, 16)).astype(np.float32)
_ws[:, 12:16] = (
    _ws[:, 12:16] * (params_5inch["w_max"] - params_5inch["w_min"])
    + params_5inch["w_min"]
)
NN_CONTROL_INPUTS = _ws


@pytest.fixture(scope="module")
def network():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
    )


@pytest.fixture(scope="module")
def network_std():
    return np.ones(4, dtype=np.float32)


@pytest.fixture(scope="module")
def env():
    return Quadcopter3DGates(
        num_envs=1,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        gates_ahead=1,
        randomization=randomization_fixed_params_5inch,
    )


@pytest.fixture(scope="module")
def lib(network, network_std, env, tmp_path_factory):
    out = tmp_path_factory.mktemp("c_code")
    generate_neural_network(
        network, env, network_std,
        params_5inch["w_min"], params_5inch["w_max"],
        str(out),
    )
    emit_controller(str(out))
    return build_library(str(out))


def _nn_forward(lib: ctypes.CDLL, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    c_in = (ctypes.c_float * len(x))(*x)
    c_out = (ctypes.c_float * 4)()
    lib.nn_forward(c_in, c_out)
    return np.array(c_out[:])


def _nn_control(lib: ctypes.CDLL, ctx: NNCtx, world_state: np.ndarray) -> np.ndarray:
    ws = np.asarray(world_state, dtype=np.float32)
    c_in = (ctypes.c_float * len(ws))(*ws)
    c_out = (ctypes.c_float * 4)()
    lib.nn_control(ctypes.byref(ctx), c_in, c_out)
    return np.array(c_out[:])


def _check_or_update(outputs: np.ndarray, golden_path: Path, update: bool):
    if update or not golden_path.exists():
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.save(golden_path, outputs)
        return
    np.testing.assert_array_equal(outputs, np.load(golden_path))


def test_nn_forward_golden(lib, request):
    update = request.config.getoption("--update-golden")
    outputs = np.array([_nn_forward(lib, x) for x in NN_FORWARD_INPUTS])
    _check_or_update(outputs, GOLDEN_DIR / "nn_forward.npy", update)


def test_nn_forward_matches_torch(lib, network):
    with torch.no_grad():
        torch_outputs = network(torch.tensor(NN_FORWARD_INPUTS)).cpu().numpy()
    c_outputs = np.array([_nn_forward(lib, x) for x in NN_FORWARD_INPUTS])
    np.testing.assert_allclose(c_outputs, torch_outputs, atol=1e-5)


def test_nn_control_golden(lib, request):
    update = request.config.getoption("--update-golden")
    ctx = lib.nn_ctx_init()
    lib.nn_set_deterministic(ctypes.byref(ctx), True)
    outputs = np.array([_nn_control(lib, ctx, ws) for ws in NN_CONTROL_INPUTS])
    _check_or_update(outputs, GOLDEN_DIR / "nn_control.npy", update)


@pytest.fixture(scope="module")
def rust_ctrl(network, network_std, env):
    generate_rust(
        network, env, network_std,
        params_5inch["w_min"], params_5inch["w_max"],
        "drone_controller",
    )
    build_rust("nn_controller")
    import nn_controller as _nc
    importlib.reload(_nc)
    return _nc.NNController()


def test_rust_control_matches_c(lib, rust_ctrl):
    """Rust and C controllers must produce bit-identical outputs on the same inputs."""
    ctx = lib.nn_ctx_init()
    lib.nn_set_deterministic(ctypes.byref(ctx), True)

    c_outputs = np.array([_nn_control(lib, ctx, ws) for ws in NN_CONTROL_INPUTS])
    rust_outputs = np.array(
        [rust_ctrl.control(ws.tolist()) for ws in NN_CONTROL_INPUTS]
    )

    np.testing.assert_allclose(rust_outputs, c_outputs, atol=1e-5)
