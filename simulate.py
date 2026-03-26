import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO

from optimal_quad_control_rl import lib_models
from optimal_quad_control_rl.quad_wpnav_env import QuadcopterWaypointNav
from optimal_quad_control_rl.quadcopter_animation import animation
from optimal_quad_control_rl.randomization import RANDOMIZATION_ALGORITHMS


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    cfg = lib_models.TrainConfig.model_validate(config.train)
    route_cfg = lib_models.WaypointRouteConfig.model_validate(config.waypoints)
    randomization = RANDOMIZATION_ALGORITHMS[cfg.randomization]

    waypoints = np.asarray(route_cfg.waypoints)
    start_pos = np.asarray(route_cfg.start_pos) if route_cfg.start_pos else None

    env = QuadcopterWaypointNav(
        num_envs=1,
        randomization=randomization,
        waypoints=waypoints,
        start_pos=start_pos,
        wps_ahead=3,
        initialize_at_random_waypoints=False,
    )

    model_path = cfg.models_dir / cfg.name / "100000000.zip"
    model = PPO.load(model_path)

    env.reset()

    def run():
        actions, _ = model.predict(env.states, deterministic=True)
        env.step(actions)
        return env.render()

    animation.view(run, fps=float(1 / env.dt), waypoints=waypoints)


if __name__ == "__main__":
    main()
