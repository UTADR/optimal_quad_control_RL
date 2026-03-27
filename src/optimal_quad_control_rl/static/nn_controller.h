#ifndef NN_CONTROLLER_H
#define NN_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

#include "neural_network.h"

extern uint8_t target_gate_index;

void nn_reset(void);
void nn_set_deterministic(bool value);
void nn_control(const float world_state[16], float motor_cmds[4]);

#endif /* NN_CONTROLLER_H */
