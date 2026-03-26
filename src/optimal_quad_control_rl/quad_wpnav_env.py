import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import VecEnv

from optimal_quad_control_rl import quad_model

_PARAMS = [
    "k_x", "k_y", "k_w",
    "k_p1", "k_p2", "k_p3", "k_p4",
    "k_q1", "k_q2", "k_q3", "k_q4",
    "k_r1", "k_r2", "k_r3", "k_r4", "k_r5", "k_r6", "k_r7", "k_r8",
    "tau", "k", "w_min", "w_max",
]


class QuadcopterWaypointNav(VecEnv):
    """Vectorized waypoint-navigation environment.

    The drone must visit a sequence of 3-D waypoints in order.  A waypoint is
    considered *reached* when the drone passes within *proximity_threshold*
    metres of it.  The episode ends when all waypoints have been reached (if
    *loop_waypoints* is False) or continues indefinitely by cycling through
    them (if True).

    Observation layout (per env, flattened over state history):
        [0:3]   pos_W  − wp_current         (position relative to current wp)
        [3:6]   vel_W                        (world-frame velocity)
        [6:9]   phi, theta, psi              (euler angles, world frame)
        [9:12]  p, q, r                      (body rates)
        [12:16] w1..w4                       (motor RPMs)
        [16 : 16+3*wps_ahead]
                wp_{k+1} − wp_k, …          (next *wps_ahead* waypoints
                                             relative to current wp, world frame)
        [16+3*wps_ahead : 16+3*wps_ahead+4*num_action_history]
                action history (optional)
        [...]   param encoding (optional)
    """

    def __init__(
        self,
        num_envs: int,
        randomization,
        waypoints: NDArray,                 # (N, 3) float32
        start_pos: NDArray | None = None,   # defaults to waypoints[0]
        wps_ahead: int = 3,
        proximity_threshold: float = 0.3,
        loop_waypoints: bool = True,
        motor_limit: float = 1.0,
        initialize_at_random_waypoints: bool = True,
        num_state_history: int = 0,
        num_action_history: int = 0,
        history_step_size: int = 1,
        param_input: bool = False,
        param_input_noise: float = 0.0,
        seed: int | None = None,
    ):
        self.seed_val = seed
        if seed is not None:
            np.random.seed(seed)

        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.num_wps = self.waypoints.shape[0]
        self.wps_ahead = wps_ahead
        self.proximity_threshold = proximity_threshold
        self.loop_waypoints = loop_waypoints
        self.start_pos = (
            np.asarray(start_pos, dtype=np.float32)
            if start_pos is not None
            else self.waypoints[0].copy()
        )
        self.initialize_at_random_waypoints = initialize_at_random_waypoints

        def rand_f(n):
            param_dict = randomization(n)
            return np.array([param_dict[p] for p in _PARAMS]).T

        self.randomization = rand_f
        self.params = self.randomization(num_envs)

        self.motor_limit = motor_limit
        self.num_state_history = num_state_history
        self.num_action_history = num_action_history
        self.history_step_size = history_step_size
        self.param_input = param_input
        self.param_input_noise = param_input_noise

        if self.param_input:
            params_noisy = self.params * np.random.uniform(
                1 - param_input_noise, 1 + param_input_noise, size=self.params.shape
            )
            self.param_encoding = quad_model.param_encoding(params_noisy)

        # observation / action spaces
        u_lim = 2 * motor_limit - 1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(4,))

        self.state_len = (
            16
            + 3 * wps_ahead
            + 4 * num_action_history
            + 9 * param_input
        )
        self.obs_len = self.state_len * (1 + num_state_history)
        observation_space = spaces.Box(
            low=np.full(self.obs_len, -np.inf, dtype=np.float32),
            high=np.full(self.obs_len, np.inf, dtype=np.float32),
        )

        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # running buffers
        self.world_states = np.zeros((num_envs, 16), dtype=np.float32)
        self.states = np.zeros((num_envs, self.obs_len), dtype=np.float32)

        num_hist = 40
        self.state_hist = np.zeros((num_envs, num_hist, self.state_len), dtype=np.float32)
        self.action_hist = np.zeros((num_envs, num_hist, 4), dtype=np.float32)

        self.max_steps = 1200
        self.dt = np.float32(0.01)

        self.target_wps = np.zeros(num_envs, dtype=int)
        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)

        # axis-aligned bounding box of waypoints + margin for out-of-bounds check
        margin = 3.0
        self._wp_min = self.waypoints.min(axis=0) - margin
        self._wp_max = self.waypoints.max(axis=0) + margin

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_states(self):
        wp = self.waypoints[self.target_wps % self.num_wps]

        new_states = np.zeros((self.num_envs, self.state_len), dtype=np.float32)

        # position relative to current waypoint (world frame, no rotation)
        new_states[:, 0:3] = self.world_states[:, 0:3] - wp

        # velocity (world frame)
        new_states[:, 3:6] = self.world_states[:, 3:6]

        # attitude and rates (body frame, unchanged)
        new_states[:, 6:16] = self.world_states[:, 6:16]

        # lookahead: wp[k+1] - wp[k] for k = target, target+1, ...
        for i in range(self.wps_ahead):
            idx_curr = self.target_wps % self.num_wps
            idx_next = (self.target_wps + i + 1) % self.num_wps
            new_states[:, 16 + 3 * i : 16 + 3 * i + 3] = (
                self.waypoints[idx_next] - self.waypoints[idx_curr]
            )

        # action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:, 0] = self.actions
        for i in range(self.num_action_history):
            base = 16 + 3 * self.wps_ahead + 4 * i
            new_states[:, base : base + 4] = (
                self.action_hist[:, (i + 1) * self.history_step_size - 1]
            )

        if self.param_input:
            new_states[:, 16 + 3 * self.wps_ahead + 4 * self.num_action_history :] = (
                self.param_encoding
            )

        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:, 0] = new_states
        self.states = self.state_hist[
            :,
            0 : (self.num_state_history + 1) * self.history_step_size : self.history_step_size,
        ].reshape((self.num_envs, -1))

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset_(self, dones: NDArray) -> NDArray:
        num_reset = dones.sum()

        if self.initialize_at_random_waypoints:
            self.target_wps[dones] = np.random.randint(0, self.num_wps, size=num_reset)
            wp = self.waypoints[self.target_wps[dones] % self.num_wps]
            x0, y0, z0 = wp[:, 0], wp[:, 1], wp[:, 2]
        else:
            self.target_wps[dones] = 0
            x0 = np.full(num_reset, self.start_pos[0])
            y0 = np.full(num_reset, self.start_pos[1])
            z0 = np.full(num_reset, self.start_pos[2])

        vx0 = np.random.uniform(-0.5, 0.5, size=(num_reset,))
        vy0 = np.random.uniform(-0.5, 0.5, size=(num_reset,))
        vz0 = np.random.uniform(-0.5, 0.5, size=(num_reset,))

        phi0   = np.random.uniform(-np.pi / 9, np.pi / 9, size=(num_reset,))
        theta0 = np.random.uniform(-np.pi / 9, np.pi / 9, size=(num_reset,))
        psi0   = np.random.uniform(-np.pi,     np.pi,     size=(num_reset,))

        p0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))
        q0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))
        r0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))

        w0 = np.random.uniform(-1, 1, size=(num_reset, 4))

        self.world_states[dones] = np.stack(
            [x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0,
             w0[:, 0], w0[:, 1], w0[:, 2], w0[:, 3]],
            axis=1,
        )
        self.step_counts[dones] = 0
        self.params[dones] = self.randomization(num_reset)

        if self.param_input:
            params_noisy = self.params[dones] * np.random.uniform(
                1 - self.param_input_noise,
                1 + self.param_input_noise,
                size=self.params[dones].shape,
            )
            self.param_encoding[dones] = quad_model.param_encoding(params_noisy)

        self._update_states()
        return self.states

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

    def step_wait(self):
        new_states = self.world_states + self.dt * quad_model.f_func(
            self.world_states, self.actions, self.params
        )
        self.step_counts += 1

        pos_old = self.world_states[:, 0:3]
        pos_new = new_states[:, 0:3]
        wp = self.waypoints[self.target_wps % self.num_wps]

        # progress reward: reduction in distance to current waypoint
        d2wp_old = np.linalg.norm(pos_old - wp, axis=1)
        d2wp_new = np.linalg.norm(pos_new - wp, axis=1)
        rate_penalty = 0.001 * np.linalg.norm(new_states[:, 9:12], axis=1)
        rewards = (d2wp_old - d2wp_new) - rate_penalty

        # waypoint reached
        wp_reached = d2wp_new < self.proximity_threshold
        self.target_wps[wp_reached] += 1
        if self.loop_waypoints:
            self.target_wps %= self.num_wps

        # termination conditions
        ground_collision = new_states[:, 2] > 0
        rewards[ground_collision] = -10.0

        out_of_bounds = np.any(
            (new_states[:, 0:3] < self._wp_min) | (new_states[:, 0:3] > self._wp_max),
            axis=1,
        )
        out_of_bounds |= np.any(np.abs(new_states[:, 9:12]) > 1000, axis=1)
        rewards[out_of_bounds] = -10.0

        all_wps_reached = ~self.loop_waypoints & (self.target_wps >= self.num_wps)
        rewards[all_wps_reached] = 10.0

        max_steps_reached = self.step_counts >= self.max_steps

        dones = ground_collision | out_of_bounds | max_steps_reached | all_wps_reached
        self.dones = dones

        self.world_states = new_states
        self.reset_(dones)

        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["wp_reached"] = wp_reached[i]
            infos[i]["all_wps_reached"] = all_wps_reached[i]

        return self.states, rewards, dones, infos

    def render(self, mode="human"):
        state_dict = dict(
            zip(
                ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi",
                 "p", "q", "r", "w1", "w2", "w3", "w4"],
                self.world_states.T,
            )
        )
        action_dict = dict(
            zip(["u1", "u2", "u3", "u4"], (np.array(self.actions.T) + 1) / 2)
        )
        return {**state_dict, **action_dict}

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_attr(self, attr_name, indices=None):
        raise AttributeError()

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
