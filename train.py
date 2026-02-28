# library imports
import math
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

import hydra
import numpy as np
import rich
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from optimal_quad_control_rl import lib_models

# custom imports
from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates
from optimal_quad_control_rl.quadcopter_animation import animation
from optimal_quad_control_rl.randomization import RANDOMIZATION_ALGORITHMS


def train(model, test_env, log_name, models_dir, video_log_dir, env, n=int(1e8)):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps * env.num_envs * 10
    while model.num_timesteps < n:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=log_name,
        )
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir / log_name / str(time_steps))
        print("Model saved at", models_dir / log_name / str(time_steps))
        # save policy animation
        # animate_policy(
        #     model,
        #     test_env,
        #     record_steps=1200,
        #     record_file=video_log_dir + '/' + log_name + '/' + str(time_steps) + '.mp4',
        #     show_window=False
        # )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    cfg = lib_models.TrainConfig.model_validate(config.train)
    # print summary of the arguments
    rich.print(cfg)

    # DEFINE RACE TRACK
    track_cfg = lib_models.RaceTrackConfig.model_validate(config.track)
    randomization = RANDOMIZATION_ALGORITHMS[cfg.randomization]
    env = Quadcopter3DGates(
        num_envs=100,
        gates_pos=np.asarray(track_cfg.gate_pos),
        gate_yaw=np.asarray(track_cfg.gate_yaw),
        start_pos=np.asarray(track_cfg.start_pos),
        randomization=randomization,
        gates_ahead=1,
        num_state_history=cfg.state_history,
        num_action_history=cfg.action_history,
        history_step_size=cfg.history_step_size,
        param_input=cfg.param_input,
        param_input_noise=cfg.param_input_noise,
    )
    test_env = Quadcopter3DGates(
        num_envs=1,
        gates_pos=np.asarray(track_cfg.gate_pos),
        gate_yaw=np.asarray(track_cfg.gate_yaw),
        start_pos=np.asarray(track_cfg.start_pos),
        randomization=randomization,
        gates_ahead=1,
        num_state_history=cfg.state_history,
        num_action_history=cfg.action_history,
        history_step_size=cfg.history_step_size,
        param_input=cfg.param_input,
        param_input_noise=cfg.param_input_noise,
    )

    # Wrap the environment in a Monitor wrapper
    env = VecMonitor(env)

    # MODEL DEFINITION
    # policy_kwcfg = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64,64], vf=[64,64])], log_std_init = 0)
    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"pi": cfg.pi, "vf": cfg.vf},
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

    # ANIMATION FUNCTION
    # def animate_policy(model, env, deterministic=False, log_times=False, print_vel=False, log=None, **kwcfg):
    #     env.reset()
    #     def run():
    #         actions, _ = model.predict(env.states, deterministic=deterministic)

    #         # print('actions=', actions)
    #         # print('states=', env.states)
    #         # print('')

    #         states, rewards, dones, infos = env.step(actions)
    #         if log != None:
    #             log(states)
    #         if print_vel:
    #             # compute mean velocity
    #             vels = env.world_states[:,3:6]
    #             mean_vel = np.linalg.norm(vels, axis=1).mean()
    #             print(mean_vel)
    #         if log_times:
    #             if rewards[0] == 10:
    #                 print(env.step_counts[0]*env.dt)

    #         return env.render()
    #     animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, **kwcfg)

    # animate untrained policy (use this to set the recording camera position)
    # animate_policy(model, test_env)

    # TESTING
    test_env.reset()

    # do 100 steps and print state and action
    # for i in range(100):
    #     print("step", i)
    #     num = test_env.num_state_history + 1
    #     state_len = int(len(test_env.states[0]) / num)
    #     for j in range(num):
    #         print(
    #             "state", j, "=", test_env.states[0][j * state_len : (j + 1) * state_len]
    #         )
    #     actions, _ = model.predict(test_env.states, deterministic=True)
    #     states, rewards, dones, infos = test_env.step(actions)
    #     print("actions=", actions[0])
    #     print("")

    # TRAINING
    # training loop saves model every 10 policy rollouts and saves a video animation
    # name = 'figure8_64_64_again!'
    # import shutil
    # shutil.rmtree(log_dir + '/' + name + '_0', ignore_errors=True)
    # shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
    # shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)

    # RUN TRAINING LOOP
    name = cfg.name

    # check if model already exists
    # if os.path.exists(models_dir + '/' + name):
    #     print(f"Model {name} already exists. Do you want to overwrite it (this will delete the existing model/logs/videos)? (y/n)")

    import shutil

    if os.path.exists(cfg.log_dir / f"{name}_0"):
        print("Deleting logs...")
        shutil.rmtree(cfg.log_dir / f"{name}_0", ignore_errors=True)
    if os.path.exists(cfg.models_dir / name):
        print("Deleting models...")
        shutil.rmtree(cfg.models_dir / name, ignore_errors=True)
    if os.path.exists(cfg.video_log_dir / name):
        print("Deleting videos...")
        shutil.rmtree(cfg.video_log_dir / name, ignore_errors=True)

    print("Training model", name)
    train(model, test_env, name, cfg.models_dir, cfg.video_log_dir, env)


if __name__ == "__main__":
    main()
