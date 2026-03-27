// Weights, deployment constants, and nn_forward — written by export.py, built via build.rs.
include!(concat!(env!("OUT_DIR"), "/generated.rs"));

pub struct ADRController {
    pub target_gate: u8,
}

impl ADRController {
    pub fn new() -> Self {
        Self { target_gate: 0 }
    }

    pub fn reset(&mut self) {
        self.target_gate = 0;
    }

    /// Run one control step.
    ///
    /// `world_state`: 16-element slice `[x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r, w1-w4]`
    ///   where `w1-w4` are physical RPMs.
    /// `on_output`: called with the raw NN output before clamping; use `|_| {}` for deterministic.
    ///
    /// Returns normalised motor commands in `[0, 1]`.
    pub fn control<F: FnMut(&mut [f32])>(
        &mut self,
        world_state: &[f32; 16],
        mut on_output: F,
    ) -> [f32; 4] {
        let pos = [world_state[0], world_state[1], world_state[2]];
        let vel = [world_state[3], world_state[4], world_state[5]];
        let yaw = world_state[8];

        let mut target_pos = GATE_POS[self.target_gate as usize];
        let mut target_yaw = GATE_YAW[self.target_gate as usize];

        // Advance gate index when drone crosses the gate plane.
        let dot = libm::cosf(target_yaw) * (pos[0] - target_pos[0])
            + libm::sinf(target_yaw) * (pos[1] - target_pos[1]);
        if dot > 0.0 {
            self.target_gate = (self.target_gate + 1) % NUM_GATES as u8;
            target_pos = GATE_POS[self.target_gate as usize];
            target_yaw = GATE_YAW[self.target_gate as usize];
        }

        let (c, s) = (libm::cosf(target_yaw), libm::sinf(target_yaw));

        let pos_rel = [
            c * (pos[0] - target_pos[0]) + s * (pos[1] - target_pos[1]),
            -s * (pos[0] - target_pos[0]) + c * (pos[1] - target_pos[1]),
            pos[2] - target_pos[2],
        ];
        let vel_rel = [c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1], vel[2]];
        let mut yaw_rel = yaw - target_yaw;
        while yaw_rel > core::f32::consts::PI {
            yaw_rel -= 2.0 * core::f32::consts::PI;
        }
        while yaw_rel < -core::f32::consts::PI {
            yaw_rel += 2.0 * core::f32::consts::PI;
        }

        let mut nn_input = [0f32; NN_INPUT_SIZE];
        nn_input[0..3].copy_from_slice(&pos_rel);
        nn_input[3..6].copy_from_slice(&vel_rel);
        nn_input[6] = world_state[6];
        nn_input[7] = world_state[7];
        nn_input[8] = yaw_rel;
        nn_input[9] = world_state[9];
        nn_input[10] = world_state[10];
        nn_input[11] = world_state[11];
        for k in 0..4 {
            nn_input[12 + k] = (world_state[12 + k] - W_MIN) * 2.0 / (W_MAX - W_MIN) - 1.0;
        }
        for i in 0..GATES_AHEAD {
            let idx = (self.target_gate as usize + i + 1) % NUM_GATES;
            nn_input[16 + 4 * i] = GATE_POS_REL[idx][0];
            nn_input[16 + 4 * i + 1] = GATE_POS_REL[idx][1];
            nn_input[16 + 4 * i + 2] = GATE_POS_REL[idx][2];
            nn_input[16 + 4 * i + 3] = GATE_YAW_REL[idx];
        }

        let mut nn_output = nn_forward(&nn_input);
        on_output(&mut nn_output);

        let mut motor_cmds = [0f32; NN_OUTPUT_SIZE];
        for i in 0..NN_OUTPUT_SIZE {
            let v = nn_output[i].clamp(-1.0, U_MAX);
            motor_cmds[i] = (v + 1.0) / 2.0;
        }
        motor_cmds
    }
}

impl Default for ADRController {
    fn default() -> Self {
        Self::new()
    }
}
