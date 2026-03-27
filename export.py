import argparse
import os.path
import shutil

import numpy as np

from optimal_quad_control_rl import export
from optimal_quad_control_rl.quad_race_env import (
    Quadcopter3DGates,
    gate_pos,
    gate_yaw,
    start_pos,
)
from optimal_quad_control_rl.quadcopter_animation import animation
from optimal_quad_control_rl.randomization import (
    params_5inch,
    randomization_fixed_params_5inch,
)


def main():
    parser = argparse.ArgumentParser(description="Export trained PPO policy to C")
    parser.add_argument("model_path", help="Path to trained model .zip")
    parser.add_argument(
        "--out-dir", default="c_code", help="Output directory for C files"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Run animated C-controller simulation"
    )
    args = parser.parse_args()

    w_min_n = params_5inch["w_min"]
    w_max_n = params_5inch["w_max"]

    _, network, network_std = export.load_network(args.model_path)
    print("Network:", network)
    print("Output std:", network_std)

    test_env = Quadcopter3DGates(
        num_envs=1,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        gates_ahead=1,
        randomization=randomization_fixed_params_5inch,
    )

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    nn_src, nn_hdr = export.generate_neural_network(
        network, test_env, network_std, w_min_n, w_max_n, args.out_dir
    )
    ctrl_src, ctrl_hdr = export.emit_controller(args.out_dir)
    print(f"Generated {nn_src}, {nn_hdr}")
    print(f"Copied    {ctrl_src}, {ctrl_hdr}")

    lib = export.build_library(args.out_dir)

    x = np.random.rand(20)
    fmt = lambda arr: [round(float(v), 5) for v in arr]
    print("c_network:    ", fmt(export.c_network(lib, x)))
    print("torch_network:", fmt(export.torch_network_infer(network, x)))
    print("nn_control_c: ", fmt(export.nn_control_c(lib, x, w_min_n, w_max_n)))

    if args.simulate:
        test_env.reset()
        lib.nn_reset()
        crashes: list[bool] = []
        action_list: list[np.ndarray] = []

        def run():
            world_state = test_env.world_states[0]
            action = export.nn_control_c(lib, world_state.copy(), w_min_n, w_max_n)
            action_list.append(action)
            steps = test_env.step_counts[0] + 1
            states, _, dones, _ = test_env.step(np.array([action]))
            if dones[0]:
                crash = steps != test_env.max_steps
                crashes.append(crash)
                print("crash" if crash else "success")
                lib.nn_reset()
            if len(crashes) == 100:
                print(f"Crash rate: {np.mean(crashes):.2%}")
            return test_env.render()

        animation.view(run, gate_pos=test_env.gate_pos, gate_yaw=test_env.gate_yaw)


if __name__ == "__main__":
    main()
