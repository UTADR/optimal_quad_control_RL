#ifndef NN_CONTROLLER_H
#define NN_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

#include "neural_network.h"

typedef struct nn_ctx {
  uint8_t target_gate_index;
  bool deterministic;
} nn_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif

nn_ctx_t nn_ctx_init(void);
void nn_reset(nn_ctx_t* ctx);
void nn_set_deterministic(nn_ctx_t* ctx, bool value);
void nn_control(nn_ctx_t* ctx, const float world_state[16],
                float motor_cmds[4]);

#ifdef __cplusplus
}
#endif

#endif /* NN_CONTROLLER_H */
