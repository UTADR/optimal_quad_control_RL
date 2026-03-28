import argparse

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
    parser = argparse.ArgumentParser(description="Export trained PPO policy to Rust")
    parser.add_argument("model_path", help="Path to trained model .zip")
    parser.add_argument(
        "--simulate", action="store_true", help="Run animated controller simulation"
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

    rs_path = export.generate_rust(
        network, test_env, network_std, w_min_n, w_max_n, "drone_controller"
    )
    print(f"Generated {rs_path}")
    export.build_rust("nn_controller")
    print("Rust extension built.")

    import nn_controller  # available after maturin develop

    ctrl = nn_controller.NNController()
    world_state = [0.0] * 16
    fmt = lambda arr: [round(float(v), 5) for v in arr]
    print("rust_control (zero state):", fmt(ctrl.control(world_state)))

    if args.simulate:
        test_env.reset()
        ctrl.reset()
        crashes: list[bool] = []

        def run():
            ws = test_env.world_states[0].copy()
            ws[12:16] = (ws[12:16] + 1) / 2 * (w_max_n - w_min_n) + w_min_n
            motor_cmds = ctrl.control(ws.tolist())
            action = np.array(motor_cmds) * 2 - 1
            steps = test_env.step_counts[0] + 1
            _, _, dones, _ = test_env.step(np.array([action]))
            if dones[0]:
                crash = steps != test_env.max_steps
                crashes.append(crash)
                print("crash" if crash else "success")
                ctrl.reset()
            if len(crashes) == 100:
                print(f"Crash rate: {np.mean(crashes):.2%}")
            return test_env.render()

        animation.view(run, gate_pos=test_env.gate_pos, gate_yaw=test_env.gate_yaw)


if __name__ == "__main__":
    main()
