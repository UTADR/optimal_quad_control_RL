# optimal_quad_control_RL

Reinforcement learning for time optimal end-to-end quadcopter control

https://arxiv.org/abs/2504.21586

## Prerequisites

We converted `optimal_quad_control_RL` to be managed by the `uv` package
manager.

On a debian-based system, you can run the following commands to get `uv`:

```bash
sudo apt-get install pipx
pipx ensurepath
pipx install uv
```

> [!TIP]
>
> You can inspect the project dependencies under the `dependencies` section of
> `pyproject.toml`.

Then, run the main train script with:

```bash
uv run train.py 5inch_drone run0_5inch_10_percent --randomization 5inch_30_percent
```

Finally, run the visualization script:

```bash
uv run simulate.py
```

> [!NOTE]
>
> At the time of writing, the simulation script hard-codes the path to the
> trained model, forcing `5inch_drone` as the session name,
> `run0_5inch_10_percent` as the drone name, and `5inch_30_percent` as the
> randomization name. You can change these values in `simulate.py` if you want
> to visualize a different model.

---
