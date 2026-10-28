# volume_wgpu

`volume_wgpu` is a small Python package for scalar increment operations on 3D microscopy volumes using `wgpu-py`. The package keeps the GPU wrapper minimal, validates NumPy inputs carefully, and provides a NumPy reference implementation that is independent of the GPU path.

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

For a regular installed package build:

```bash
python -m pip install .
```

## Supported inputs

- Volumes are NumPy arrays with shape `(z, y, x)`.
- Data must be `float32` and three-dimensional, nonempty, and C-contiguous after conversion.
- Noncontiguous arrays are accepted by making a contiguous upload copy internally.
- `x` varies fastest in memory order; this is the standard contiguous layout for `numpy.ndarray` with `order='C'`.
- The GPU path currently supports a scalar increment only; no file I/O, convolution, plugins, or async execution are included.

## Demo

Run the included smoke test:

```bash
python examples/smoke_test.py
```

The demo constructs a synthetic volume, runs the GPU increment operation, compares against the NumPy reference, prints adapter/backend details, and reports success.

## Tests

Run the full test suite:

```bash
pytest -q
```

Run only the CPU/reference checks:

```bash
pytest -q -m "not gpu"
```

Run the GPU tests when a compatible adapter/device is available:

```bash
pytest -q -m gpu
```

Strict mode fails instead of skipping when GPU execution is required:

```bash
WGPU_REQUIRE_GPU=1 pytest -q -m gpu
```

## GPU requirements and behavior

- GPU tests are skipped only when a usable `wgpu` adapter/device cannot be initialized.
- A software Vulkan backend is acceptable for smoke testing and should be described explicitly as software execution, not as hardware validation.
- `WGPU_REQUIRE_GPU=1` turns that skip into a hard failure so CI or developer workflows can enforce GPU execution.

## Project structure

```text
src/
  volume_wgpu/
    __init__.py
    context.py
    gpu/
      __init__.py
      increment.py
    reference/
      __init__.py
      increment.py
    shaders/
      __init__.py
      increment.wgsl
```

## Extending the package

To add another GPU wrapper:

1. Add a new operation module under `src/volume_wgpu/gpu/`.
2. Create a tiny validation helper and reuse the `GPUContext` object.
3. Package any WGSL shader in `src/volume_wgpu/shaders/` and load it with `importlib.resources`.
4. Add a NumPy reference under `src/volume_wgpu/reference/`.
5. Write a comparison test in `tests/test_<operation>.py` that checks the GPU result against the reference implementation.

## CI and build checks

The repository includes CI that:

- installs the package with its dev dependencies,
- runs the CPU and reference tests,
- builds the distributable package, and
- exercises the installed package outside the source tree to confirm packaged shader resources load correctly.

## Current implementation status

This repository currently supports:

- scalar `float32` increment operations on nonempty 3D NumPy volumes,
- a reusable `GPUContext`,
- a minimal `Incrementer` GPU wrapper,
- a NumPy reference implementation,
- smoke tests for GPU and CPU pathways,
- validation tests for malformed inputs.

Remaining limitations:

- no file I/O or visualization,
- no asynchronous or tiled execution,
- no general plugin system,
- no support for non-float32 or non-3D arrays.
