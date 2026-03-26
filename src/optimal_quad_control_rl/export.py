import ctypes
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates


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


def generate_neural_network(network: nn.Sequential, output_dir: str):
    source_path = os.path.join(output_dir, "neural_network.c")
    header_path = os.path.join(output_dir, "neural_network.h")

    with open(source_path, "w") as f:
        f.write('#include "neural_network.h"\n')
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n\n")

        i = 1
        for layer in network:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.cpu().numpy()
                biases = layer.bias.data.cpu().numpy()
                f.write(f"const float weights_fc{i}[] = {{\n")
                f.write(
                    ",\n".join([", ".join(map(float_to_str, row)) for row in weights])
                )
                f.write("\n};\n\n")
                f.write(f"const float biases_fc{i}[] = {{\n")
                f.write(", ".join(map(float_to_str, biases)))
                f.write("\n};\n\n")
                i += 1

        f.write(
            "void nn_linear(const float* weights, const float* biases, const float* input,"
            " int in_features, int out_features, float* output) {\n"
        )
        f.write("    for (int i = 0; i < out_features; ++i) {\n")
        f.write("        float neuron = biases[i];\n")
        f.write("        for (int j = 0; j < in_features; ++j) {\n")
        f.write("            neuron += input[j] * weights[i * in_features + j];\n")
        f.write("        }\n")
        f.write("        output[i] = neuron;\n")
        f.write("    }\n")
        f.write("}\n\n")

        f.write("void nn_relu(float* input, int size) {\n")
        f.write("    for (int i = 0; i < size; ++i) {\n")
        f.write("        input[i] = fmaxf(0, input[i]);\n")
        f.write("    }\n")
        f.write("}\n\n")

        f.write("void nn_tanh(float* input, int size) {\n")
        f.write("    for (int i = 0; i < size; ++i) {\n")
        f.write("        input[i] = tanh(input[i]);\n")
        f.write("    }\n")
        f.write("}\n\n")

        f.write("void nn_forward(const float* input, float* output) {\n")
        layer_size = network[0].out_features
        num_linear = sum(isinstance(it, nn.Linear) for it in network)
        i, input_array = 0, "input"
        for layer in network:
            if isinstance(layer, nn.Linear):
                i += 1
                if i < num_linear:
                    f.write(f"    float fc{i}_output[{layer.out_features}];\n")
                    f.write(
                        f"    nn_linear(weights_fc{i}, biases_fc{i}, {input_array},"
                        f" {layer.in_features}, {layer.out_features}, fc{i}_output);\n"
                    )
                    input_array = f"fc{i}_output"
                else:
                    f.write(
                        f"    nn_linear(weights_fc{i}, biases_fc{i}, {input_array},"
                        f" {layer.in_features}, {layer.out_features}, output);\n"
                    )
                    input_array = "output"
                layer_size = layer.out_features
            elif isinstance(layer, nn.ReLU):
                f.write(f"    nn_relu({input_array}, {layer_size});\n")
            elif isinstance(layer, nn.Tanh):
                f.write(f"    nn_tanh({input_array}, {layer_size});\n")
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
        f.write("}\n")

    with open(header_path, "w") as f:
        f.write("#ifndef NEURAL_NETWORK_H\n")
        f.write("#define NEURAL_NETWORK_H\n\n")
        f.write("void nn_forward(const float* input, float* output);\n")
        f.write("\n#endif // NEURAL_NETWORK_H\n")

    return source_path, header_path


def generate_controller(
    network_std: np.ndarray,
    test_env: Quadcopter3DGates,
    w_min_n: float,
    w_max_n: float,
    output_dir: str,
):
    name = "nn_controller"
    source_path = os.path.join(output_dir, f"{name}.c")
    header_path = os.path.join(output_dir, f"{name}.h")

    num_gates = test_env.num_gates
    gates_ahead = test_env.gates_ahead
    motor_lim = test_env.motor_limit
    u_max = 2 * motor_lim - 1

    with open(header_path, "w") as f:
        f.write(f"#ifndef {name.upper()}_H\n")
        f.write(f"#define {name.upper()}_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdbool.h>\n\n")
        f.write(f"#define GATES_AHEAD {gates_ahead}\n")
        f.write(f"#define NUM_GATES {num_gates}\n\n")
        f.write('#include "neural_network.h"\n\n')
        f.write("extern const float gate_pos[NUM_GATES][3];\n")
        f.write("extern const float gate_yaw[NUM_GATES];\n")
        f.write("extern const float start_pos[3];\n")
        f.write("extern const float start_yaw;\n")
        f.write("extern uint8_t target_gate_index;\n\n")
        f.write("void nn_reset(void);\n")
        f.write("void nn_set_deterministic(bool value);\n")
        f.write(
            "void nn_control(const float world_state[16], float motor_cmds[4]);\n\n"
        )
        f.write("#endif\n")

    with open(source_path, "w") as f:
        f.write(f'#include "{name}.h"\n')
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n\n")
        f.write("bool deterministic = false;\n\n")

        f.write("const float output_std[4] = {\n")
        for v in network_std:
            f.write(f"    {v},\n")
        f.write("};\n\n")

        f.write("const float gate_pos[NUM_GATES][3] = {\n")
        for i in range(num_gates):
            p = test_env.gate_pos[i]
            f.write(f"    {{{p[0]}, {p[1]}, {p[2]}}},\n")
        f.write("};\n\n")

        f.write("const float gate_yaw[NUM_GATES] = {\n")
        for i in range(num_gates):
            f.write(f"    {test_env.gate_yaw[i]},\n")
        f.write("};\n\n")

        sp = test_env.start_pos
        f.write(f"const float start_pos[3] = {{{sp[0]}, {sp[1]}, {sp[2]}}};\n")
        f.write(f"const float start_yaw = {test_env.gate_yaw[0]};\n\n")

        f.write("const float gate_pos_rel[NUM_GATES][3] = {\n")
        for i in range(num_gates):
            p = test_env.gate_pos_rel[i]
            f.write(f"    {{{p[0]}, {p[1]}, {p[2]}}},\n")
        f.write("};\n\n")

        f.write("const float gate_yaw_rel[NUM_GATES] = {\n")
        for i in range(num_gates):
            f.write(f"    {test_env.gate_yaw_rel[i]},\n")
        f.write("};\n\n")

        f.write("uint8_t target_gate_index = 0;\n\n")

        f.write("void nn_reset(void) {\n")
        f.write("    target_gate_index = 0;\n")
        f.write("}\n\n")
        f.write("void nn_set_deterministic(bool value) {\n")
        f.write("    deterministic = value;\n")
        f.write("}\n\n")

        f.write("void nn_control(const float world_state[16], float motor_cmds[4]) {\n")
        f.write(
            "    float pos[3] = {world_state[0], world_state[1], world_state[2]};\n"
        )
        f.write(
            "    float vel[3] = {world_state[3], world_state[4], world_state[5]};\n"
        )
        f.write("    float yaw = world_state[8];\n\n")
        f.write(
            "    float target_pos[3] = {gate_pos[target_gate_index][0],"
            " gate_pos[target_gate_index][1], gate_pos[target_gate_index][2]};\n"
        )
        f.write("    float target_yaw = gate_yaw[target_gate_index];\n\n")
        f.write(
            "    if (cosf(target_yaw) * (pos[0] - target_pos[0])"
            " + sinf(target_yaw) * (pos[1] - target_pos[1]) > 0) {\n"
        )
        f.write("        target_gate_index = (target_gate_index + 1) % NUM_GATES;\n")
        f.write("        target_pos[0] = gate_pos[target_gate_index][0];\n")
        f.write("        target_pos[1] = gate_pos[target_gate_index][1];\n")
        f.write("        target_pos[2] = gate_pos[target_gate_index][2];\n")
        f.write("        target_yaw = gate_yaw[target_gate_index];\n")
        f.write("    }\n\n")
        f.write("    float pos_rel[3] = {\n")
        f.write(
            "        cosf(target_yaw)*(pos[0]-target_pos[0]) + sinf(target_yaw)*(pos[1]-target_pos[1]),\n"
        )
        f.write(
            "        -sinf(target_yaw)*(pos[0]-target_pos[0]) + cosf(target_yaw)*(pos[1]-target_pos[1]),\n"
        )
        f.write("        pos[2] - target_pos[2]\n")
        f.write("    };\n")
        f.write("    float vel_rel[3] = {\n")
        f.write("        cosf(target_yaw)*vel[0] + sinf(target_yaw)*vel[1],\n")
        f.write("        -sinf(target_yaw)*vel[0] + cosf(target_yaw)*vel[1],\n")
        f.write("        vel[2]\n")
        f.write("    };\n")
        f.write("    float yaw_rel = yaw - target_yaw;\n")
        f.write("    while (yaw_rel >  M_PI) { yaw_rel -= 2*M_PI; }\n")
        f.write("    while (yaw_rel < -M_PI) { yaw_rel += 2*M_PI; }\n\n")
        f.write("    float nn_input[16+4*GATES_AHEAD];\n")
        f.write("    for (int i = 0; i < 3; i++) {\n")
        f.write("        nn_input[i]   = pos_rel[i];\n")
        f.write("        nn_input[i+3] = vel_rel[i];\n")
        f.write("    }\n")
        f.write("    nn_input[6]  = world_state[6];\n")
        f.write("    nn_input[7]  = world_state[7];\n")
        f.write("    nn_input[8]  = yaw_rel;\n")
        f.write("    nn_input[9]  = world_state[9];\n")
        f.write("    nn_input[10] = world_state[10];\n")
        f.write("    nn_input[11] = world_state[11];\n")
        f.write(f"    const float w_min = {w_min_n}, w_max = {w_max_n};\n")
        for k in range(4):
            f.write(
                f"    nn_input[{12 + k}] = (world_state[{12 + k}] - w_min) * 2 / (w_max - w_min) - 1;\n"
            )
        f.write("    for (int i = 0; i < GATES_AHEAD; i++) {\n")
        f.write("        uint8_t index = (target_gate_index + i + 1) % NUM_GATES;\n")
        f.write("        nn_input[16+4*i]   = gate_pos_rel[index][0];\n")
        f.write("        nn_input[16+4*i+1] = gate_pos_rel[index][1];\n")
        f.write("        nn_input[16+4*i+2] = gate_pos_rel[index][2];\n")
        f.write("        nn_input[16+4*i+3] = gate_yaw_rel[index];\n")
        f.write("    }\n\n")
        f.write("    float nn_output[4];\n")
        f.write("    nn_forward(nn_input, nn_output);\n\n")
        f.write("    if (!deterministic) {\n")
        f.write("        for (int i = 0; i < 4; i++) {\n")
        f.write("            float u1 = (float)rand() / RAND_MAX;\n")
        f.write("            float u2 = (float)rand() / RAND_MAX;\n")
        f.write(
            "            nn_output[i] += output_std[i] * sqrtf(-2*logf(u1)) * cosf(2*M_PI*u2);\n"
        )
        f.write("        }\n")
        f.write("    }\n\n")
        f.write("    for (int i = 0; i < 4; i++) {\n")
        f.write(f"        if (nn_output[i] > {u_max}) nn_output[i] = {u_max};\n")
        f.write("        if (nn_output[i] < -1) nn_output[i] = -1;\n")
        f.write("        motor_cmds[i] = (nn_output[i] + 1) / 2;\n")
        f.write("    }\n")
        f.write("}\n")

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
