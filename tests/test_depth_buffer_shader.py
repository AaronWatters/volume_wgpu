from __future__ import annotations

import os

import numpy as np
import pytest

from volume_wgpu import GPUContext
from volume_wgpu.buffer import StorageBuffer, UniformBuffer


@pytest.fixture(scope="session")
def gpu_context():
    try:
        return GPUContext()
    except Exception as exc:  # pragma: no cover - exercised by pytest skip policy
        if os.environ.get("WGPU_REQUIRE_GPU") == "1":
            pytest.fail(f"WGPU_REQUIRE_GPU=1 but GPU init failed: {exc}")
        pytest.skip(f"GPU is unavailable in this environment: {exc}")


@pytest.mark.gpu
def test_depth_buffer_shader_extent(gpu_context):
    output = StorageBuffer(gpu_context, f32size=4)
    geometry = UniformBuffer(gpu_context, f32size=6)
    geometry.write(np.array([100.0, 100.0, 3.0, 4.0, 1.0, 1.0], dtype=np.float32))
    shader_module = gpu_context.load_shader("depth_buffer.wgsl", "depth_buffer_test.wgsl")
    layout = gpu_context.device.create_bind_group_layout(
        entries=[
            output.bind_group_layout_entry(0),
            geometry.bind_group_layout_entry(1),
        ]
    )
    pipeline = gpu_context.device.create_compute_pipeline(
        layout=gpu_context.device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": shader_module, "entry_point": "depth_buffer_extent_test"},
    )
    bind_group = gpu_context.device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": output.buffer}},
            {"binding": 1, "resource": {"buffer": geometry.buffer}},
        ],
    )
    encoder = gpu_context.device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1)
    compute_pass.end()
    gpu_context.device.queue.submit([encoder.finish()])

    np.testing.assert_array_equal(output.read(), np.array([1.0, 1.0, 3.0, 4.0], dtype=np.float32))
