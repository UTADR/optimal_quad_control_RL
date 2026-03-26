import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from stable_baselines3 import PPO

from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


def float_to_str(x):
    return str(float(x))


def load_network(model_path: str):
    model = PPO.load(model_path)
    network = nn.Sequential(
        *list(model.policy.mlp_extractor.policy_net),
        model.policy.action_net,
    )
    network_std = model.policy.log_std.exp().cpu().detach().numpy()
    return model, network, network_std


def _build_nn_context(network: nn.Sequential) -> dict:
    linear_layers = []
    i = 1
    for layer in network:
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.cpu().numpy()
            biases = layer.bias.data.cpu().numpy()
            linear_layers.append({
                "idx": i,
                "weights_str": ",\n".join(", ".join(map(float_to_str, row)) for row in weights),
                "biases_str": ", ".join(map(float_to_str, biases)),
            })
            i += 1

    forward_steps = []
    layer_size = network[0].out_features
    num_linear = sum(isinstance(l, nn.Linear) for l in network)
    i, input_array = 0, "input"
    for layer in network:
        if isinstance(layer, nn.Linear):
            i += 1
            if i < num_linear:
                forward_steps.append(f"float fc{i}_output[{layer.out_features}];")
                forward_steps.append(
                    f"nn_linear(weights_fc{i}, biases_fc{i}, {input_array},"
                    f" {layer.in_features}, {layer.out_features}, fc{i}_output);"
                )
                input_array = f"fc{i}_output"
            else:
                forward_steps.append(
                    f"nn_linear(weights_fc{i}, biases_fc{i}, {input_array},"
                    f" {layer.in_features}, {layer.out_features}, output);"
                )
            layer_size = layer.out_features
        elif isinstance(layer, nn.ReLU):
            forward_steps.append(f"nn_relu({input_array}, {layer_size});")
        elif isinstance(layer, nn.Tanh):
            forward_steps.append(f"nn_tanh({input_array}, {layer_size});")
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    return {"linear_layers": linear_layers, "forward_steps": forward_steps}


def generate_neural_network(network: nn.Sequential, output_dir: str):
    ctx = _build_nn_context(network)
    source_path = os.path.join(output_dir, "neural_network.c")
    header_path = os.path.join(output_dir, "neural_network.h")
    Path(source_path).write_text(_env.get_template("neural_network.c.j2").render(ctx))
    Path(header_path).write_text(_env.get_template("neural_network.h.j2").render())
    return source_path, header_path


def generate_controller(
    network_std: np.ndarray,
    test_env: Quadcopter3DGates,
    w_min_n: float,
    w_max_n: float,
    output_dir: str,
):
    name = "nn_controller"
    motor_lim = test_env.motor_limit
    ctx = {
        "name": name,
        "num_gates": test_env.num_gates,
        "gates_ahead": test_env.gates_ahead,
        "output_std": [float(v) for v in network_std],
        "gate_pos": [[float(v) for v in p] for p in test_env.gate_pos],
        "gate_yaw": [float(v) for v in test_env.gate_yaw],
        "start_pos": [float(v) for v in test_env.start_pos],
        "start_yaw": float(test_env.gate_yaw[0]),
        "gate_pos_rel": [[float(v) for v in p] for p in test_env.gate_pos_rel],
        "gate_yaw_rel": [float(v) for v in test_env.gate_yaw_rel],
        "w_min": w_min_n,
        "w_max": w_max_n,
        "u_max": 2 * motor_lim - 1,
    }
    source_path = os.path.join(output_dir, f"{name}.c")
    header_path = os.path.join(output_dir, f"{name}.h")
    Path(source_path).write_text(_env.get_template("nn_controller.c.j2").render(ctx))
    Path(header_path).write_text(_env.get_template("nn_controller.h.j2").render(ctx))
    return source_path, header_path


def build_library(output_dir: str) -> ctypes.CDLL:
    abs_dir = os.path.abspath(output_dir)
    subprocess.run("gcc -fPIC -c *.c", shell=True, cwd=abs_dir, check=True)
    subprocess.run(
        "gcc -shared -Wl,-soname,libtools.so -o libtools.so *.o",
        shell=True,
        cwd=abs_dir,
        check=True,
    )
    subprocess.run("rm *.o", shell=True, cwd=abs_dir, check=True)
    lib = ctypes.CDLL(os.path.join(abs_dir, "libtools.so"))
    lib.nn_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.nn_set_deterministic.argtypes = [ctypes.c_bool]
    lib.nn_set_deterministic.restype = None
    lib.nn_control.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def c_network(lib: ctypes.CDLL, x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    c_in = (ctypes.c_float * len(x))(*x)
    c_out = (ctypes.c_float * 4)()
    lib.nn_forward(c_in, c_out)
    return np.clip(np.array(c_out[:]), -1, 1)


def torch_network_infer(network: nn.Sequential, x: np.ndarray) -> np.ndarray:
    t = torch.tensor(x, dtype=torch.float32)
    return np.clip(network(t).cpu().detach().numpy(), -1, 1)


def nn_control_c(
    lib: ctypes.CDLL, x: np.ndarray, w_min_n: float, w_max_n: float
) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    x[12:16] = (x[12:16] + 1) / 2 * (w_max_n - w_min_n) + w_min_n
    c_in = (ctypes.c_float * len(x))(*x)
    c_out = (ctypes.c_float * 4)()
    lib.nn_control(c_in, c_out)
    return (np.array(c_out[:]) * 2) - 1
