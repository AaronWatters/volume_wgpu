import pytest
from numpy.testing import assert_array_equal

from volume_wgpu.reference.depth_buffer import DepthBuffer, DepthGeometry


def test_geometry_converts_between_coordinates_and_offsets():
    geometry = DepthGeometry(width=3, height=2, x0=10.0, y0=20.0, dx=2.0, dy=5.0)

    assert geometry.X_to_L((14.0, 25.0, 7.0)) == (2, 1, 7.0)
    assert geometry.L_to_X((2, 1, 7.0)) == (14.0, 25.0, 7.0)
    assert geometry.L__offset((2, 1, 0.0)) == 5
    assert geometry.offset_to_L(5) == (2, 1, 0)


def test_depth_buffer_stores_data_and_depth_at_coordinates():
    buffer = DepthBuffer(DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0))
    coordinate = (1.0, 0.0, 3.5)

    assert buffer.get(coordinate) == 0.0
    assert buffer.get_depth(coordinate) == float("inf")

    buffer.set(coordinate, 42.0)
    buffer.set_depth(coordinate, 3.5)

    assert buffer.get(coordinate) == 42.0
    assert buffer.get_depth(coordinate) == 3.5


def test_depth_buffer_loads_rows_into_data_and_depth_buffers():
    buffer = DepthBuffer(DepthGeometry(width=3, height=2, x0=5.0, y0=-2.0, dx=2.0, dy=4.0))

    buffer.load_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    buffer.load_depth([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

    assert_array_equal(buffer.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_array_equal(buffer.depth, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


@pytest.mark.parametrize("operation", ["get", "set", "get_depth", "set_depth"])
def test_depth_buffer_rejects_out_of_bounds_coordinates(operation):
    buffer = DepthBuffer(DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0))
    method = getattr(buffer, operation)
    arguments = ((2.0, 0.0, 0.0), 1.0) if operation in {"set", "set_depth"} else ((2.0, 0.0, 0.0),)

    with pytest.raises(ValueError, match="out of bounds"):
        method(*arguments)


@pytest.mark.parametrize("loader", ["load_data", "load_depth"])
def test_depth_buffer_rejects_mismatched_array_dimensions(loader):
    buffer = DepthBuffer(DepthGeometry(width=2, height=2, x0=0.0, y0=0.0, dx=1.0, dy=1.0))

    with pytest.raises(ValueError, match="dimensions"):
        getattr(buffer, loader)([[1.0, 2.0]])