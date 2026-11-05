
import numpy as np
import wgpu

from ..context import GPUContext
from ..buffer import StorageBuffer, UniformBuffer


class DepthBufferGeometry:
    """Wrapper for GPU resident depth buffer geometry data stored in a uniform buffer."""

    default_binding = 2

    def __init__(
            self, 
            context: GPUContext,
            width: float,
            height: float,
            x0: float = 0.0,
            y0: float = 0.0,
            dx: float = 1.0,
            dy: float = 1.0,):
        self.context = context
        self.buffer = UniformBuffer(context, f32size=6)
        self.width = int(width)
        self.height = int(height)
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy

    def get_numpy_array(self) -> np.ndarray:
        """Return the geometry data as a numpy array."""
        return np.array([self.width, self.height, self.x0, self.y0, self.dx, self.dy], dtype=np.float32)

    def write(self) -> None:
        """Write the geometry data to the GPU buffer."""
        self.buffer.write(self.get_numpy_array())

    def bind_group_layout_entry(self, binding: int = None) -> dict[str, any]:
        """Return a dictionary suitable for use in a bind group layout entry."""
        if binding is None:
            binding = self.default_binding
        return self.buffer.bind_group_layout_entry(binding)

    def binding_group_entry(self, binding: int = None) -> dict[str, any]:
        """Return a dictionary suitable for use in a bind group entry."""
        if binding is None:
            binding = self.default_binding
        return self.buffer.binding_group_entry(binding)


class DepthBuffer:
    """Wrapper for GPU resident depth buffer data stored
    a storage buffer for the data.
    a storage buffer for the depth values.
    a uniform buffer for the geometry data.
    """

    def init(self, context: GPUContext, geometry: DepthBufferGeometry):
        self.context = context
        self.geometry = geometry
        self.f32size = int(geometry.width * geometry.height)
        self.data_buffer = StorageBuffer(context, f32size=self.f32size)
        self.depth_buffer = StorageBuffer(context, f32size=self.f32size)
        self.my_layout = self.binding_group_layout()
        self.my_group = self.binding_group(layout=self.my_layout)

    def binding_group_layout(self, with_geometry: bool = True):
        device = self.context.device
        entries = [
            self.data_buffer.bind_group_layout_entry(0),
            self.depth_buffer.bind_group_layout_entry(1),
        ]
        if with_geometry:
            entries.append(self.geometry.bind_group_layout_entry(2))
        return device.create_bind_group_layout(entries=entries)

    def binding_group(self, layout=None, with_geometry: bool = True):
        device = self.context.device
        entries = [
            self.data_buffer.binding_group_entry(0),
            self.depth_buffer.binding_group_entry(1),
        ]
        if with_geometry:
            entries.append(self.geometry.binding_group_entry(2))
        if layout is None:
            layout = self.binding_group_layout(with_geometry=with_geometry)
        return device.create_bind_group(layout=layout, entries=entries)

    def write_data(self, data: np.ndarray) -> None:
        """Write the data values to the GPU buffer."""
        self.data_buffer.write(data)

    def write_depth(self, depth: np.ndarray) -> None:
        """Write the depth values to the GPU buffer."""
        self.depth_buffer.write(depth)

    def init_constants(self, data_value, depth_value):
        """Initialize the data and depth buffers with constant values."""
        self.data_buffer.write_constant(data_value)
        self.depth_buffer.write_constant(depth_value)

    def reshape_buffer_array(self, buffer_array: np.ndarray) -> np.ndarray:
        """Reshape a 1d buffer array to a 2d array with the correct dimensions."""
        return buffer_array.reshape((self.geometry.height, self.geometry.width))

    def get_data_numpy_array(self, reshape: bool = True) -> np.ndarray:
        """Return the data values as a numpy array."""
        data = self.data_buffer.read()
        if reshape:
            data = self.reshape_buffer_array(data)
        return data

    def get_depth_numpy_array(self, reshape: bool = True) -> np.ndarray:
        """Return the depth values as a numpy array."""
        depth = self.depth_buffer.read()
        if reshape:
            depth = self.reshape_buffer_array(depth)
        return depth

    def copy_buffers(self, to_other: "DepthBuffer", encoder=None) -> None:
        """Copy this depth buffer's data and depth buffers to another depth buffer."""
        if self.context is not to_other.context:
            raise ValueError("Both depth buffers must use the same GPU context.")
        size = self.data_buffer.bytesize
        my_encoder = encoder
        if encoder is None:
            my_encoder = self.context.device.create_command_encoder()
        my_encoder.copy_buffer_to_buffer(
                self.data_buffer.buffer,
            0,
            to_other.data_buffer.buffer,
            0,
            size,
        )
        my_encoder.copy_buffer_to_buffer(
            self.depth_buffer.buffer,
            0,
            to_other.depth_buffer.buffer,
            0,
            size,
        )
        if encoder is None:
            self.context.device.queue.submit([my_encoder.finish()])


def create_depth_buffer(
    context: GPUContext,
    geometry: DepthBufferGeometry,
    data: np.ndarray,
    depth: np.ndarray,
) -> DepthBuffer:
    """Create and initialize a GPU depth buffer from 2D float32 arrays."""
    width = int(geometry.width)
    height = int(geometry.height)
    data = np.ascontiguousarray(data, dtype=np.float32)
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    expected_shape = (height, width)
    if data.shape != expected_shape or depth.shape != expected_shape:
        raise ValueError(f"Data and depth arrays must have shape {expected_shape}.")
    geometry.write()
    result = DepthBuffer()
    result.init(context, geometry)
    result.write_data(data.ravel())
    result.write_depth(depth.ravel())
    return result