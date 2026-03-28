# Rust Controller Workflow

## Crate layout

```
drone_controller/   — #![no_std] ADRController + NN math + generated weights
  src/
    lib.rs          — re-exports ADRController, OUTPUT_STD
    nn.rs           — linear / relu / tanh primitives (nalgebra, libm)
    utils.rs        — Box::clamp_unproject helper
    controller.rs   — gate traversal, feature extraction, control<F: FnMut(&mut [f32])>
    generated_stub.rs  — zeroed placeholder; used by cargo check / rust-analyzer
    generated.rs    — written by export.py; gitignored

nn_controller/      — std, PyO3, rand; thin wrapper over drone_controller
  src/lib.rs        — NNController pyclass; injects Box-Muller noise via on_output callback
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
Resume from a checkpoint by passing `model_path=<zip>` (see `train.py --help`).

---

## 2. Generate and build the Rust extension

```bash
uv run export.py models/<run>/model.zip
```

Default target is `rust`. This:
1. Renders `drone_controller/src/generated.rs` from `templates/generated.rs.j2`
   — bakes column-major weights, gate geometry, RPM bounds, and the `nn_forward`
     free function as an `impl`-less function in the same module.
2. Runs `maturin develop --release` inside `nn_controller/`.
3. Imports `nn_controller` and prints a smoke-test inference.

To regenerate C firmware files instead:

```bash
uv run export.py models/<run>/model.zip --target c --out-dir c_code
```

---

## 3. Simulate

```bash
uv run export.py models/<run>/model.zip --simulate
```

Runs the Rerun visualiser driven by the Rust controller.
Motor commands flow as: `ADRController::control(&ws, |_| {})` (deterministic) →
`[0,1]` → rescaled to `[-1,1]` before `env.step()`.

The `--target c --simulate` path uses the compiled `libtools.so` via ctypes instead.

---

## 4. Test codegen soundness

```bash
uv run pytest tests/test_export_golden.py
```

| Test | What it checks |
|---|---|
| `test_nn_forward_matches_torch` | C `nn_forward` ≡ PyTorch (atol 1e-5) on 1000 inputs |
| `test_nn_forward_golden` | C `nn_forward` bit-stable across runs |
| `test_nn_control_golden` | C `nn_control` (deterministic, gate-stateful) bit-stable |
| `test_rust_control_matches_c` | Rust `ADRController` ≡ C `nn_control` (atol 1e-5) on 500 world states |

`test_rust_control_matches_c` re-generates `generated.rs` from the seeded test
network, rebuilds the extension, and compares step-for-step with the C library.

Regenerate golden records after an intentional architecture change:

```bash
uv run pytest tests/test_export_golden.py --update-golden
```

---

## Weight layout

PyTorch stores weights row-major `[out, in]`.
`generate_rust()` transposes before writing: `.T.flatten()` → column-major,
so `SMatrixView::<f32, OUT, IN>::from_slice(&WEIGHTS_FCn_DATA)` is a zero-copy
view and nalgebra's GEMV iterates contiguous columns.

The C codegen writes row-major and `nn_linear` indexes `weights[i][j]` directly.
Both are correct; neither transposes at runtime.
