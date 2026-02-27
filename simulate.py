from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from optimal_quad_control_rl.quad_race_env import Quadcopter3DGates
from optimal_quad_control_rl.randomization import *
from quadcopter_animation import animation


env = Quadcopter3DGates(
    num_envs=1,
    initialize_at_random_gates=False,
    randomization=randomization_fixed_params_5inch,
    pause_if_collision=False
)

model_path = 'models/my_session/my_model_name/10000000.zip'
model = PPO.load(model_path)

def animate_policy(model, env, reset_func=None, **kwargs):
    def run():
        actions, _ = model.predict(env.states, deterministic=True)
        states, rewards, dones, infos = env.step(actions)
        out = env.render()

        return out
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, fps=1/env.dt, **kwargs)


animate_policy(model, env)
