#include "nn_controller.h"

#include <math.h>
#include <stdlib.h>

nn_ctx_t nn_ctx_init(void) {
  nn_ctx_t ctx = {0, false};
  return ctx;
}

void nn_reset(nn_ctx_t* ctx) { ctx->target_gate_index = 0; }

void nn_set_deterministic(nn_ctx_t* ctx, bool value) {
  ctx->deterministic = value;
}

void nn_control(nn_ctx_t* ctx, const float world_state[16],
                float motor_cmds[4]) {
  const float* pos = &world_state[0];
  const float* vel = &world_state[3];
  float yaw = world_state[8];

  float target_pos[3] = {
      gate_pos[ctx->target_gate_index][0],
      gate_pos[ctx->target_gate_index][1],
      gate_pos[ctx->target_gate_index][2],
  };
  float target_yaw = gate_yaw[ctx->target_gate_index];

  /* Advance gate index when drone crosses the gate plane */
  if (cosf(target_yaw) * (pos[0] - target_pos[0]) +
          sinf(target_yaw) * (pos[1] - target_pos[1]) >
      0.0f) {
    ctx->target_gate_index = (ctx->target_gate_index + 1) % NUM_GATES;
    target_pos[0] = gate_pos[ctx->target_gate_index][0];
    target_pos[1] = gate_pos[ctx->target_gate_index][1];
    target_pos[2] = gate_pos[ctx->target_gate_index][2];
    target_yaw = gate_yaw[ctx->target_gate_index];
  }

  float c = cosf(target_yaw), s = sinf(target_yaw);

  float pos_rel[3] = {
      c * (pos[0] - target_pos[0]) + s * (pos[1] - target_pos[1]),
      -s * (pos[0] - target_pos[0]) + c * (pos[1] - target_pos[1]),
      pos[2] - target_pos[2],
  };
  float vel_rel[3] = {
      c * vel[0] + s * vel[1],
      -s * vel[0] + c * vel[1],
      vel[2],
  };
  float yaw_rel = yaw - target_yaw;
  while (yaw_rel > (float)M_PI) {
    yaw_rel -= 2.0f * (float)M_PI;
  }
  while (yaw_rel < -(float)M_PI) {
    yaw_rel += 2.0f * (float)M_PI;
  }

  float nn_input[NN_INPUT_SIZE];
  for (int i = 0; i < 3; i++) {
    nn_input[i] = pos_rel[i];
    nn_input[i + 3] = vel_rel[i];
  }
  nn_input[6] = world_state[6];
  nn_input[7] = world_state[7];
  nn_input[8] = yaw_rel;
  nn_input[9] = world_state[9];
  nn_input[10] = world_state[10];
  nn_input[11] = world_state[11];
  for (int k = 0; k < 4; k++) {
    nn_input[12 + k] =
        (world_state[12 + k] - W_MIN) * 2.0f / (W_MAX - W_MIN) - 1.0f;
  }
  for (int i = 0; i < GATES_AHEAD; i++) {
    uint8_t idx = (ctx->target_gate_index + i + 1) % NUM_GATES;
    nn_input[16 + 4 * i] = gate_pos_rel[idx][0];
    nn_input[16 + 4 * i + 1] = gate_pos_rel[idx][1];
    nn_input[16 + 4 * i + 2] = gate_pos_rel[idx][2];
    nn_input[16 + 4 * i + 3] = gate_yaw_rel[idx];
  }

  float nn_output[NN_OUTPUT_SIZE];
  nn_forward(nn_input, nn_output);

  if (!ctx->deterministic) {
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
      float u1 = (float)rand() / (float)RAND_MAX;
      float u2 = (float)rand() / (float)RAND_MAX;
      nn_output[i] += output_std[i] * sqrtf(-2.0f * logf(u1)) *
                      cosf(2.0f * (float)M_PI * u2);
    }
  }

  for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
    if (nn_output[i] > U_MAX) {
      nn_output[i] = U_MAX;
    }
    if (nn_output[i] < -1.0f) {
      nn_output[i] = -1.0f;
    }
    motor_cmds[i] = (nn_output[i] + 1.0f) / 2.0f;
  }
}
