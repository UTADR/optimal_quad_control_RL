import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from optimal_quad_control_rl.export import (
    build_rust,
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

# Fixed inputs — seeded, never change
_rng = np.random.default_rng(42)
NN_FORWARD_INPUTS = _rng.random((1000, 20)).astype(np.float32)
# world_state: indices 12-15 must be in physical RPM range
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


def test_rust_forward_matches_torch(rust_ctrl, network):
    """nn_forward_raw must match PyTorch inference within float32 tolerance."""
    import nn_controller as _nc

    with torch.no_grad():
        torch_outputs = network(torch.tensor(NN_FORWARD_INPUTS)).cpu().numpy()
    rust_outputs = np.array([_nc.nn_forward_raw(x.tolist()) for x in NN_FORWARD_INPUTS])

    np.testing.assert_allclose(rust_outputs, torch_outputs, atol=1e-5)
