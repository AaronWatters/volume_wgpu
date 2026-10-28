from __future__ import annotations

import numpy as np
import pytest
import wgpu
from numpy.testing import assert_allclose
from volume_wgpu import GPUContext, Incrementer
from volume_wgpu.reference.increment import increment_reference


@pytest.fixture(scope="session")
def gpu_context():
    try:
        context = GPUContext()
    except Exception as exc:  # pragma: no cover - exercised by pytest skip policy
        if __import__("os").environ.get("WGPU_REQUIRE_GPU") == "1":
            pytest.fail(f"WGPU_REQUIRE_GPU=1 but GPU init failed: {exc}")
        pytest.skip(f"GPU is unavailable in this environment: {exc}")
    return context


def _assert_increment_matches(volume: np.ndarray, increment: float, *, context: GPUContext | None = None) -> None:
    operation = Incrementer(shape=volume.shape, context=context)
    expected = increment_reference(volume, np.float32(increment))
    actual = operation.run(volume, increment=np.float32(increment))
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_gpu_small_deterministic_volume(gpu_context):
    volume = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        dtype=np.float32,
    )
    _assert_increment_matches(volume, 1.25, context=gpu_context)


@pytest.mark.gpu
def test_gpu_one_element_volume(gpu_context):
    volume = np.array([[[42.0]]], dtype=np.float32)
    _assert_increment_matches(volume, -3.5, context=gpu_context)


@pytest.mark.gpu
def test_gpu_non_divisible_by_workgroup_size(gpu_context):
    volume = np.arange(3 * 5 * 17, dtype=np.float32).reshape(3, 5, 17)
    _assert_increment_matches(volume, 0.5, context=gpu_context)


@pytest.mark.gpu
def test_gpu_positive_negative_zero_increments(gpu_context):
    for increment in (0.0, -2.5, 1.25):
        volume = np.linspace(-3.0, 3.0, 5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
        _assert_increment_matches(volume, increment, context=gpu_context)


@pytest.mark.gpu
def test_gpu_random_finite_values(gpu_context):
    rng = np.random.default_rng(1234)
    volume = rng.normal(0.0, 2.0, size=(4, 3, 5)).astype(np.float32)
    _assert_increment_matches(volume, 0.125, context=gpu_context)


@pytest.mark.gpu
def test_gpu_repeated_calls_same_instance(gpu_context):
    operation = Incrementer(shape=(2, 3, 4), context=gpu_context)
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.5) + 1.0
    first_result = operation.run(first, increment=np.float32(2.0))
    second_result = operation.run(second, increment=np.float32(-0.75))
    assert_allclose(first_result, increment_reference(first, np.float32(2.0)), rtol=1e-6, atol=1e-6)
    assert_allclose(second_result, increment_reference(second, np.float32(-0.75)), rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_gpu_noncontiguous_input(gpu_context):
    base = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    volume = base[:, ::2, 1:-1]
    _assert_increment_matches(volume, 3.0, context=gpu_context)


@pytest.mark.gpu
def test_gpu_input_preservation_and_output_independence(gpu_context):
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    original = volume.copy()
    operation = Incrementer(shape=volume.shape, context=gpu_context)
    result_a = operation.run(volume, increment=2.0)
    result_b = operation.run(volume, increment=-1.5)
    assert_allclose(volume, original, rtol=0.0, atol=0.0)
    result_a[0, 0, 0] = -999.0
    assert result_b[0, 0, 0] != -999.0


@pytest.mark.gpu
def test_gpu_input_buffer_preservation_after_execution(gpu_context):
    volume = np.arange(6 * 7 * 9, dtype=np.float32).reshape(6, 7, 9)
    operation = Incrementer(shape=volume.shape, context=gpu_context)
    operation.run(volume, increment=np.float32(5.0))
    staging = operation.device.create_buffer(size=volume.size * np.dtype(np.float32).itemsize, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    encoder = operation.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(operation.input_buffer, 0, staging, 0, staging.size)
    operation.device.queue.submit([encoder.finish()])
    staging.map_sync(wgpu.MapMode.READ)
    copied = np.frombuffer(staging.read_mapped(), dtype=np.float32, count=volume.size)
    staging.unmap()
    assert_allclose(copied.reshape(volume.shape), volume, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "shape, dtype, dim_error, increment",
    [
        ((0, 3, 4), np.float32, ValueError, 1.0),
        ((3, 0, 4), np.float32, ValueError, 1.0),
        ((3, 3, 0), np.float32, ValueError, 1.0),
        ((2, 3), np.float32, ValueError, 1.0),
        ((2, 3, 4), np.float64, TypeError, 1.0),
        ((2, 3, 4), np.float32, ValueError, np.array([1.0], dtype=np.float32)),
    ],
)
def test_reference_validation_errors(shape, dtype, dim_error, increment):
    volume = np.zeros(shape, dtype=dtype)
    if shape != (2, 3, 4):
        volume = np.zeros(shape, dtype=dtype)
    if shape == (2, 3):
        volume = np.zeros(shape, dtype=dtype)
    if shape == (2, 3, 4) and dtype == np.float64:
        volume = np.zeros(shape, dtype=dtype)
    if shape == (2, 3, 4) and isinstance(increment, np.ndarray):
        with pytest.raises(TypeError):
            increment_reference(volume, increment)
        return
    with pytest.raises(dim_error):
        increment_reference(volume, increment)


def test_reference_matches_expected_values():
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    expected = volume + np.float32(1.75)
    actual = increment_reference(volume, np.float32(1.75))
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_reference_rejects_nonscalar_increment():
    volume = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        increment_reference(volume, np.array([1.0], dtype=np.float32))
