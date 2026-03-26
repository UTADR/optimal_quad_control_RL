# Pydantic models governing the training and evaluation of the optimal quad control RL project.
# Copyright © 2026 UTADR
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field, model_validator


class TrainConfig(BaseModel):
    session_name: str = Field(..., description="Session Name")
    name: str = Field(..., description="Model name")
    pi: tuple[int, int] = Field(default=(64, 64), description="Policy architecture")
    vf: tuple[int, int] = Field(
        default=(64, 64), description="Value function architecture"
    )
    state_history: int = Field(default=0, description="State history input length")
    action_history: int = Field(default=0, description="Action history input length")
    history_step_size: int = Field(default=1, description="History step size")
    param_input: bool = Field(default=False, description="Parameter input")
    param_input_noise: float = Field(default=0.0, description="Parameter input noise")
    randomization: str = Field(default="randomized", description="Randomization")

    @computed_field
    @property
    def models_dir(self) -> Path:
        return Path.cwd() / "models" / self.session_name

    @computed_field
    @property
    def log_dir(self) -> Path:
        return Path.cwd() / "logs" / self.session_name

    @computed_field
    @property
    def video_log_dir(self) -> Path:
        return Path.cwd() / "videos" / self.session_name

    @model_validator(mode="after")
    def create_dirs(self):
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        if not self.video_log_dir.exists():
            self.video_log_dir.mkdir(parents=True)
        return self


class WaypointRouteConfig(BaseModel):
    waypoints: list[tuple[float, float, float]] = Field(
        ..., description="Ordered 3-D waypoint positions"
    )
    start_pos: tuple[float, float, float] | None = Field(
        default=None, description="Start position (defaults to first waypoint)"
    )


class RaceTrackYawUnit(str, Enum):
    radians = "radians"
    degrees = "degrees"
    multiples_pi_2 = "multiples_pi_2"


class RaceTrackConfig(BaseModel):
    gate_pos: list[tuple[float, float, float]] = Field(
        ..., description="Gate positions"
    )
    gate_yaw: list[float] = Field(..., description="Gate yaws")
    gate_yaw_unit: RaceTrackYawUnit = Field(
        default=RaceTrackYawUnit.radians, description="Gate yaw unit"
    )
    start_pos: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Start position"
    )

    @model_validator(mode="after")
    def convert_gate_yaw(self):
        if self.gate_yaw_unit == RaceTrackYawUnit.degrees:
            self.gate_yaw = [math.radians(yaw) for yaw in self.gate_yaw]
        elif self.gate_yaw_unit == RaceTrackYawUnit.multiples_pi_2:
            self.gate_yaw = [yaw * math.pi / 2 for yaw in self.gate_yaw]
        return self
