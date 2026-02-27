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
uv run train.py <session_name> <model_name>
```

---
