import shutil

import hydra
import numpy as np
import rich
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import tqdm

from optimal_quad_control_rl import lib_models
from optimal_quad_control_rl.quad_wpnav_env import QuadcopterWaypointNav
from optimal_quad_control_rl.randomization import RANDOMIZATION_ALGORITHMS


def _train(model, env, models_dir, name, n: int = int(1e8)):
    timesteps_per_save = model.n_steps * env.num_envs * 10
    with tqdm(total=n, initial=model.num_timesteps, desc="Training") as pbar:
        while model.num_timesteps < n:
            prev = model.num_timesteps
            model.learn(
                total_timesteps=timesteps_per_save,
                reset_num_timesteps=False,
                tb_log_name=name,
            )
            pbar.update(model.num_timesteps - prev)
            model.save(models_dir / name / str(model.num_timesteps))


def _make_env(route_cfg, cfg, randomization, num_envs):
    return QuadcopterWaypointNav(
        num_envs=num_envs,
        randomization=randomization,
        waypoints=np.asarray(route_cfg.waypoints),
        start_pos=np.asarray(route_cfg.start_pos) if route_cfg.start_pos else None,
        wps_ahead=3,
        num_state_history=cfg.state_history,
        num_action_history=cfg.action_history,
        history_step_size=cfg.history_step_size,
        param_input=cfg.param_input,
        param_input_noise=cfg.param_input_noise,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    cfg = lib_models.TrainConfig.model_validate(config.train)
    route_cfg = lib_models.WaypointRouteConfig.model_validate(config.waypoints)
    rich.print(cfg)

    randomization = RANDOMIZATION_ALGORITHMS[cfg.randomization]
    env = VecMonitor(_make_env(route_cfg, cfg, randomization, num_envs=100))

    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"pi": list(cfg.pi), "vf": list(cfg.vf)},
        "log_std_init": 0,
    }
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=str(cfg.log_dir),
        n_steps=1000,
        batch_size=5000,
        n_epochs=10,
        gamma=0.999,
        device="cpu",
    )

    name = cfg.name
    for path in (cfg.log_dir / f"{name}_0", cfg.models_dir / name, cfg.video_log_dir / name):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    _train(model, env, cfg.models_dir, name)


if __name__ == "__main__":
    main()
