from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from volume_wgpu import GPUContext
from volume_wgpu.gpu.combine_overlap import DepthBufferCombinerStage, create_depth_buffer
from volume_wgpu.gpu.depth_buffer import DepthBufferGeometry
from volume_wgpu.reference.combine_overlap import DepthBufferCombinerStage as ReferenceCombiner
from volume_wgpu.reference.depth_buffer import DepthBuffer as ReferenceDepthBuffer
from volume_wgpu.reference.depth_buffer import DepthGeometry


@pytest.fixture(scope="session")
def gpu_context():
    try:
        return GPUContext()
    except Exception as exc:
        pytest.skip(f"GPU is unavailable in this environment: {exc}")


@pytest.mark.gpu
def test_gpu_combine_overlap_matches_reference(gpu_context):
    geometry_a = DepthBufferGeometry(gpu_context, 2, 1, x0=0.0, y0=0.0, dx=2.0, dy=1.0)
    geometry_b = DepthBufferGeometry(gpu_context, 4, 1, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    data_a = np.array([[10.0, 20.0]], dtype=np.float32)
    depth_a = np.array([[6.0, 2.0]], dtype=np.float32)
    data_b = np.array([[30.0, 40.0, 50.0, 60.0]], dtype=np.float32)
    depth_b = np.array([[9.0, 8.0, 7.0, 6.0]], dtype=np.float32)
    buffer_a = create_depth_buffer(gpu_context, geometry_a, data_a, depth_a)
    buffer_b = create_depth_buffer(gpu_context, geometry_b, data_b, depth_b)

    stage = DepthBufferCombinerStage(buffer_a, buffer_b, context=gpu_context)
    stage.combine()
    actual_depth = stage.get_output().get_depth_numpy_array()
    actual_data = stage.get_output().get_data_numpy_array()

    reference_a = ReferenceDepthBuffer(DepthGeometry(2, 1, 0.0, 0.0, 2.0, 1.0))
    reference_b = ReferenceDepthBuffer(DepthGeometry(4, 1, 0.0, 0.0, 1.0, 1.0))
    reference_a.load_data(data_a)
    reference_a.load_depth(depth_a)
    reference_b.load_data(data_b)
    reference_b.load_depth(depth_b)
    reference_stage = ReferenceCombiner(reference_a, reference_b)
    reference_stage.combine()
    expected_data, expected_depth = reference_stage.get_output().get_numpy_arrays()

    assert_allclose(actual_depth, expected_depth, rtol=0.0, atol=1e-6)
    assert_allclose(actual_data, expected_data, rtol=0.0, atol=1e-6)
