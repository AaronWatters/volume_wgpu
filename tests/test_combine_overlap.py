from numpy.testing import assert_array_equal

from volume_wgpu.reference.combine_overlap import DepthBufferCombinerStage
from volume_wgpu.reference.depth_buffer import DepthBuffer, DepthGeometry


def _combine_depths(geometry_a, depths_a, geometry_b, depths_b):
    buffer_a = DepthBuffer(geometry_a)
    buffer_b = DepthBuffer(geometry_b)
    buffer_a.load_depth(depths_a)
    buffer_b.load_depth(depths_b)
    stage = DepthBufferCombinerStage(buffer_a, buffer_b)
    stage.combine()
    return stage.get_output().depth


def test_combine_overlap_same_geometry_takes_minimum_depth():
    geometry = DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0)

    depths = _combine_depths(
        geometry,
        [[4.0, 1.0], [8.0, 3.0]],
        geometry,
        [[2.0, 5.0], [9.0, 2.0]],
    )

    assert_array_equal(depths, [2.0, 1.0, 8.0, 2.0])


def test_combine_overlap_disjoint_geometry_preserves_second_buffer():
    geometry_a = DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    geometry_b = DepthGeometry(width=2, height=2, x0=10.0, y0=10.0, dx=1.0, dy=1.0)

    depths = _combine_depths(
        geometry_a,
        [[1.0, 2.0], [3.0, 4.0]],
        geometry_b,
        [[8.0, 7.0], [6.0, 5.0]],
    )

    assert_array_equal(depths, [8.0, 7.0, 6.0, 5.0])


def test_combine_overlap_shifted_geometry_updates_only_overlapping_columns():
    geometry_a = DepthGeometry(width=3, height=1, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    geometry_b = DepthGeometry(width=3, height=1, x0=1.0, y0=0.0, dx=1.0, dy=1.0)

    depths = _combine_depths(
        geometry_a,
        [[7.0, 5.0, 3.0]],
        geometry_b,
        [[8.0, 6.0, 4.0]],
    )

    assert_array_equal(depths, [5.0, 3.0, 4.0])


def test_combine_overlap_different_resolution_updates_multiple_output_pixels():
    geometry_a = DepthGeometry(width=2, height=1, x0=0.0, y0=0.0, dx=2.0, dy=1.0)
    geometry_b = DepthGeometry(width=4, height=1, x0=0.0, y0=0.0, dx=1.0, dy=1.0)

    depths = _combine_depths(
        geometry_a,
        [[6.0, 2.0]],
        geometry_b,
        [[9.0, 8.0, 7.0, 6.0]],
    )

    assert_array_equal(depths, [6.0, 6.0, 2.0, 2.0])


def test_combine_overlap_shifted_geometry_in_both_axes():
    geometry_a = DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    geometry_b = DepthGeometry(width=2, height=2, x0=1.0, y0=1.0, dx=1.0, dy=1.0)

    depths = _combine_depths(
        geometry_a,
        [[5.0, 3.0], [7.0, 1.0]],
        geometry_b,
        [[9.0, 8.0], [6.0, 4.0]],
    )

    assert_array_equal(depths, [1.0, 8.0, 6.0, 4.0])