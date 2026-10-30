from __future__ import annotations

import os

import numpy as np
import pytest

from volume_wgpu import GPUContext
from volume_wgpu.buffer import StorageBuffer


@pytest.fixture(scope="session")
def gpu_context():
    try:
        return GPUContext()
    except Exception as exc:  # pragma: no cover - exercised by pytest skip policy
        if os.environ.get("WGPU_REQUIRE_GPU") == "1":
            pytest.fail(f"WGPU_REQUIRE_GPU=1 but GPU init failed: {exc}")
        pytest.skip(f"GPU is unavailable in this environment: {exc}")


@pytest.mark.gpu
def test_storage_buffer_write_and_read_round_trip(gpu_context):
    expected = np.array([0.0, 1.25, -2.5, 42.0], dtype=np.float32)
    buffer = StorageBuffer(gpu_context, f32size=expected.size)

    buffer.write(expected)

    np.testing.assert_array_equal(buffer.read(), expected)
