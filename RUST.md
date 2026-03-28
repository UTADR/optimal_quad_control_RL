# Rust Controller Workflow

## Crate layout

```
drone_controller/   — #![no_std] ADRController + NN math + generated weights
  src/
    lib.rs          — re-exports ADRController, OUTPUT_STD, nn_forward, NN_{INPUT,OUTPUT}_SIZE
    nn.rs           — linear / relu / tanh primitives (nalgebra, libm)
    utils.rs        — BoxSpace::unproject helper
    controller.rs   — gate traversal, feature extraction, control<F: FnMut(&mut [f32])>
    generated_stub.rs  — zeroed placeholder; used by cargo check / rust-analyzer
    generated.rs    — written by export.py; gitignored

nn_controller/      — std, PyO3, rand; thin wrapper over drone_controller
  src/lib.rs        — NNController pyclass + nn_forward_raw pyfunction;
                      injects Box-Muller noise via on_output callback
```

`drone_controller` has no `rand` dep and no feature flags — it is always `no_std`.
Stochastic noise lives entirely in `nn_controller`, injected as a closure into `control()`.

---

## 1. Train

```bash
uv run train.py
```

Hydra config lives in `conf/`. Override on the CLI:

```bash
uv run train.py randomization=fixed_params_5inch training.n_steps=2048
```

Checkpoints are saved to `models/` every 10 rollouts.

---

## 2. Generate and build the Rust extension

```bash
uv run export.py models/<run>/model.zip
```

This:
1. Renders `drone_controller/src/generated.rs` from `templates/generated.rs.j2`
   — bakes column-major weights, gate geometry, RPM bounds, and the `nn_forward`
     free function into the `controller` module.
2. Runs `maturin develop --release` inside `nn_controller/`.
3. Imports `nn_controller` and prints a smoke-test inference.

---

## 3. Simulate

```bash
uv run export.py models/<run>/model.zip --simulate
```

Runs the Rerun visualiser driven by the Rust controller.
Motor commands flow as: `ADRController::control(&ws, |_| {})` (deterministic) →
`[0,1]` → rescaled to `[-1,1]` before `env.step()`.

---

## 4. Test codegen soundness

```bash
uv run pytest tests/test_export_golden.py
```

| Test | What it checks |
|---|---|
| `test_rust_forward_matches_torch` | `nn_forward_raw` ≡ PyTorch (atol 1e-5) on 1000 inputs |

`test_rust_forward_matches_torch` re-generates `generated.rs` from the seeded test
network, rebuilds `nn_controller`, calls `nn_forward_raw` on 1000 random inputs,
and compares against the same network evaluated in PyTorch.

---

## Weight layout

PyTorch stores weights row-major `[out, in]`.
`generate_rust()` transposes before writing: `.T.flatten()` → column-major,
so `SMatrixView::<f32, OUT, IN>::from_slice(&WEIGHTS_FCn_DATA)` is a zero-copy
view and nalgebra's GEMV iterates contiguous columns.
