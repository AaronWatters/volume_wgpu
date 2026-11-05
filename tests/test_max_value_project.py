import numpy as np
from numpy.testing import assert_array_equal

from volume_wgpu.reference.depth_buffer import DepthBuffer, DepthGeometry
from volume_wgpu.reference.max_value_project import MaxProjectionStage
from volume_wgpu.reference.volume import Volume, VolumeGeometry


def _stage_for(volume_data, depth_geometry):
    data = np.array(volume_data, dtype=np.float32)
    volume = Volume(
        data,
        geometry=VolumeGeometry(np.eye(4, dtype=np.float32), data.shape),
    )
    depth_buffer = DepthBuffer(depth_geometry)
    return MaxProjectionStage(from_volume=volume, to_depth_buffer=depth_buffer)


def test_execute_projects_max_value_and_depth_for_each_pixel():
    stage = _stage_for(
        [
            [[1.0, 5.0, 3.0]],
            [[2.0, 4.0, 9.0]],
        ],
        DepthGeometry(width=2, height=1, x0=0.0, y0=0.0, dx=1.0, dy=1.0),
    )

    output = stage.execute()

    assert_array_equal(output.data, [5.0, 9.0])
    assert_array_equal(output.depth, [1.0, 2.0])


def test_execute_respects_custom_initial_values_for_uncovered_pixels():
    stage = _stage_for(
        [[[2.0]]],
        DepthGeometry(width=2, height=1, x0=0.0, y0=0.0, dx=1.0, dy=1.0),
    )

    output = stage.execute(init_data=-7.0, init_depths=99.0)

    assert_array_equal(output.data, [2.0, -7.0])
    assert_array_equal(output.depth, [0.0, 99.0])


def test_run_shader_ignores_out_of_bounds_instance():
    stage = _stage_for(
        [[[3.0], [4.0]]],
        DepthGeometry(width=1, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0),
    )
    stage.to_depth_buffer.data[:] = [10.0, 20.0]
    stage.to_depth_buffer.depth[:] = [30.0, 40.0]

    nrows = stage.from_volume.geometry.nrows()
    stage.run_shader(nrows + 1)

    assert_array_equal(stage.to_depth_buffer.data, [10.0, 20.0])
    assert_array_equal(stage.to_depth_buffer.depth, [30.0, 40.0])


def test_execute_with_permuted_volume_axes_geometry():
    data = np.array(
        [
            [[1.0, 10.0], [3.0, 4.0]],
            [[5.0, 2.0], [7.0, 6.0]],
        ],
        dtype=np.float32,
    )
    transform = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    volume = Volume(data, geometry=VolumeGeometry(transform, data.shape))
    depth_buffer = DepthBuffer(DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0))
    stage = MaxProjectionStage(from_volume=volume, to_depth_buffer=depth_buffer)

    output = stage.execute()

    assert_array_equal(output.data, [5.0, 7.0, 10.0, 6.0])
    assert_array_equal(output.depth, [1.0, 1.0, 0.0, 1.0])
