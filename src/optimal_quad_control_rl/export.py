import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from stable_baselines3 import PPO

from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


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
            linear_layers.append(
                {
                    "idx": i,
                    "weights_str": ",\n".join(
                        ", ".join(str(float(v)) for v in row) for row in weights
                    ),
                    "biases_str": ", ".join(str(float(v)) for v in biases),
                }
            )
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

    linear_list = [l for l in network if isinstance(l, nn.Linear)]
    input_size = linear_list[0].in_features
    output_size = linear_list[-1].out_features

    return {
        "linear_layers": linear_layers,
        "forward_steps": forward_steps,
        "input_size": input_size,
        "output_size": output_size,
    }


def generate_neural_network(
    network: nn.Sequential,
    test_env: Quadcopter3DGates,
    network_std: np.ndarray,
    w_min_n: float,
    w_max_n: float,
    output_dir: str,
):
    """Generate neural_network.{c,h} — W&B, forward pass, and all deployment constants."""
    ctx = _build_nn_context(network)
    ctx.update(
        {
            "num_gates": test_env.num_gates,
            "gates_ahead": test_env.gates_ahead,
            "output_std": [float(v) for v in network_std],
            "gate_pos": [[float(v) for v in p] for p in test_env.gate_pos],
            "gate_yaw": [float(v) for v in test_env.gate_yaw],
            "gate_pos_rel": [[float(v) for v in p] for p in test_env.gate_pos_rel],
            "gate_yaw_rel": [float(v) for v in test_env.gate_yaw_rel],
            "start_pos": [float(v) for v in test_env.start_pos],
            "start_yaw": float(test_env.gate_yaw[0]),
            "w_min": w_min_n,
            "w_max": w_max_n,
            "u_max": float(2 * test_env.motor_limit - 1),
        }
    )
    out = Path(output_dir)
    source_path = out / "neural_network.c"
    header_path = out / "neural_network.h"
    source_path.write_text(_env.get_template("neural_network.c.j2").render(ctx))
    header_path.write_text(_env.get_template("neural_network.h.j2").render(ctx))
    return str(source_path), str(header_path)


def emit_controller(output_dir: str):
    """Copy the hand-written nn_controller.{c,h} into the output directory."""
    out = Path(output_dir)
    for name in ("nn_controller.c", "nn_controller.h"):
        shutil.copy(_STATIC_DIR / name, out / name)
    return str(out / "nn_controller.c"), str(out / "nn_controller.h")


class NNCtx(ctypes.Structure):
    _fields_ = [
        ("target_gate_index", ctypes.c_uint8),
        ("deterministic", ctypes.c_bool),
    ]


def build_library(output_dir: str) -> ctypes.CDLL:
    abs_dir = str(Path(output_dir).resolve())
    subprocess.run("gcc -fPIC -c *.c", shell=True, cwd=abs_dir, check=True)
    subprocess.run(
        "gcc -shared -Wl,-soname,libtools.so -o libtools.so *.o",
        shell=True,
        cwd=abs_dir,
        check=True,
    )
    subprocess.run("rm *.o", shell=True, cwd=abs_dir, check=True)
    lib = ctypes.CDLL(str(Path(abs_dir) / "libtools.so"))
    lib.nn_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.nn_ctx_init.restype = NNCtx
    lib.nn_reset.argtypes = [ctypes.POINTER(NNCtx)]
    lib.nn_reset.restype = None
    lib.nn_set_deterministic.argtypes = [ctypes.POINTER(NNCtx), ctypes.c_bool]
    lib.nn_set_deterministic.restype = None
    lib.nn_control.argtypes = [
        ctypes.POINTER(NNCtx),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.nn_control.restype = None
    return lib


def c_network(lib: ctypes.CDLL, x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    c_in = (ctypes.c_float * len(x))(*x)
    c_out = (ctypes.c_float * 4)()
    lib.nn_forward(c_in, c_out)
    return np.clip(np.array(c_out[:]), -1, 1)


def torch_network_infer(network: nn.Sequential, x: np.ndarray) -> np.ndarray:
    t = torch.tensor(x, dtype=torch.float32, device="cuda")
    return np.clip(network(t).cpu().detach().numpy(), -1, 1)


def nn_control_c(
    lib: ctypes.CDLL, ctx: NNCtx, x: np.ndarray, w_min_n: float, w_max_n: float
) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    x[12:16] = (x[12:16] + 1) / 2 * (w_max_n - w_min_n) + w_min_n
    c_in = (ctypes.c_float * len(x))(*x)
    c_out = (ctypes.c_float * 4)()
    lib.nn_control(ctypes.byref(ctx), c_in, c_out)
    return (np.array(c_out[:]) * 2) - 1


def _fmt_f32(v: float) -> str:
    """Format a float as a Rust f32 literal in scientific notation."""
    return f"{v:.9e}_f32"


def generate_rust(
    network: nn.Sequential,
    test_env: Quadcopter3DGates,
    network_std: np.ndarray,
    w_min_n: float,
    w_max_n: float,
    controller_dir: str = "drone_controller",
) -> str:
    """Generate drone_controller/src/generated.rs with column-major W&B and deployment constants."""
    linear_list = [layer for layer in network if isinstance(layer, nn.Linear)]
    input_size = linear_list[0].in_features
    output_size = linear_list[-1].out_features
    hidden_size = linear_list[0].out_features

    # Pair each linear layer with its following activation (relu/tanh/None).
    layer_acts: list[tuple[nn.Linear, str | None]] = []
    pending: nn.Linear | None = None
    for layer in network:
        if isinstance(layer, nn.Linear):
            if pending is not None:
                layer_acts.append((pending, None))
            pending = layer
        elif isinstance(layer, nn.ReLU):
            layer_acts.append((pending, "relu"))  # type: ignore[arg-type]
            pending = None
        elif isinstance(layer, nn.Tanh):
            layer_acts.append((pending, "tanh"))  # type: ignore[arg-type]
            pending = None
    if pending is not None:
        layer_acts.append((pending, None))

    linear_layers = []
    for i, (layer, _) in enumerate(layer_acts, start=1):
        # Column-major layout: element [row, col] at index col*ROWS + row.
        # PyTorch weight is [out, in] (row-major); .T gives [in, out]; .flatten() is column-major.
        weights_cm = layer.weight.data.cpu().numpy().T.flatten()
        biases = layer.bias.data.cpu().numpy()
        linear_layers.append(
            {
                "idx": i,
                "weight_len": len(weights_cm),
                "weights_rs": ",\n    ".join(_fmt_f32(v) for v in weights_cm),
                "bias_len": len(biases),
                "biases_rs": ", ".join(_fmt_f32(v) for v in biases),
            }
        )

    # Generate Rust forward-pass statements.
    rs_forward_steps: list[str] = []
    prev_var = "x"
    for i, (layer, act) in enumerate(layer_acts, start=1):
        is_last = i == len(layer_acts)
        out_var = "out" if is_last else f"h{i}"
        in_sz, out_sz = layer.in_features, layer.out_features
        rs_forward_steps += [
            f"let w{i} = SMatrixView::<f32, {out_sz}, {in_sz}>::from_slice(&WEIGHTS_FC{i}_DATA);",
            f"let b{i} = SVector::<f32, {out_sz}>::from_column_slice(&BIASES_FC{i}_DATA);",
        ]
        # First layer: prev_var is "x" which is already &SVector — pass directly.
        # Later layers: prev_var is an owned SVector — borrow it.
        input_ref = prev_var if i == 1 else f"&{prev_var}"
        linear_expr = f"linear(&w{i}, &b{i}, {input_ref})"
        if act == "relu":
            rs_forward_steps.append(f"let {out_var} = relu(&{linear_expr});")
        elif act == "tanh":
            rs_forward_steps.append(f"let {out_var} = tanh(&{linear_expr});")
        else:
            rs_forward_steps.append(f"let {out_var} = {linear_expr};")
        prev_var = out_var

    used_acts = sorted({act for _, act in layer_acts if act is not None})
    act_imports = ", ".join(["linear"] + used_acts)

    ctx = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "num_gates": test_env.num_gates,
        "gates_ahead": test_env.gates_ahead,
        "w_min": float(w_min_n),
        "w_max": float(w_max_n),
        "u_max": float(2 * test_env.motor_limit - 1),
        "linear_layers": linear_layers,
        "rs_forward_steps": rs_forward_steps,
        "act_imports": act_imports,
        "output_std": [_fmt_f32(v) for v in network_std],
        "gate_pos": [[float(v) for v in p] for p in test_env.gate_pos],
        "gate_yaw": [float(v) for v in test_env.gate_yaw],
        "gate_pos_rel": [[float(v) for v in p] for p in test_env.gate_pos_rel],
        "gate_yaw_rel": [float(v) for v in test_env.gate_yaw_rel],
        "start_pos": [float(v) for v in test_env.start_pos],
        "start_yaw": float(test_env.gate_yaw[0]),
    }

    out_path = Path(controller_dir) / "src" / "generated.rs"
    out_path.write_text(_env.get_template("generated.rs.j2").render(ctx))
    return str(out_path)


def build_rust(nn_controller_dir: str = "nn_controller") -> None:
    """Build and install nn_controller via maturin develop --release."""
    subprocess.run(
        ["maturin", "develop", "--release"],
        cwd=str(Path(nn_controller_dir).resolve()),
        check=True,
    )
