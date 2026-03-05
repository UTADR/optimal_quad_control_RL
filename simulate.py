from omegaconf import DictConfig
from stable_baselines3 import PPO
from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates
from optimal_quad_control_rl import lib_models, randomization
from optimal_quad_control_rl.quadcopter_animation import animation

import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    cfg = lib_models.TrainConfig.model_validate(config.train)
    randomizer = randomization.RANDOMIZATION_ALGORITHMS[cfg.randomization]

    env = Quadcopter3DGates(
        num_envs=1,
        initialize_at_random_gates=False,
        randomization=randomizer,
        pause_if_collision=False
    )

    model_path = cfg.models_dir / cfg.name / "100000000.zip"
    model = PPO.load(model_path)

    def animate_policy(model, env, reset_func=None, **kwargs):
        def run():
            actions, _ = model.predict(env.states, deterministic=True)
            states, rewards, dones, infos = env.step(actions)
            out = env.render()

            return out
        animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, fps=1/env.dt, **kwargs)


    animate_policy(model, env)

if __name__ == '__main__':
    main()

