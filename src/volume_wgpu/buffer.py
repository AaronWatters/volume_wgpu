"""
Wrappers for different GPU buffer types, input, output, and uniform.
For simplicity all buffers are float32, which is sufficient for the intended use cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import wgpu

from .context import GPUContext

f32bytes = np.dtype(np.float32).itemsize

class Buffer:
    """Base class for GPU buffers."""
    def __init__(self, context: GPUContext, f32size: int, usage: wgpu.BufferUsage):
        self.context = context
        self.device = context.device
        self.f32size = f32size
        self.bytesize = f32size * f32bytes
        self.usage = usage
        self.buffer = self.device.create_buffer(
            size=self.bytesize,
            usage=usage,
        )
    def binding_group_entry(self, binding: int) -> dict[str, Any]:
        """Return a dictionary suitable for use in a bind group entry."""
        return {
            "binding": binding,
            "resource": {"buffer": self.buffer, "offset": 0, "size": self.bytesize},
        }
    def bind_group_layout_entry(self, binding: int, type: str) -> dict[str, Any]:
        """Return a dictionary suitable for use in a bind group layout entry."""
        return {
            "binding": binding,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": type},
        }
    def write(self, data: np.ndarray) -> None:
        """Write data to the GPU buffer."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if data.dtype != np.float32:
            # convert to float32 if not already
            data = data.astype(np.float32)
        if data.size != self.f32size:
            raise ValueError(f"Data size {data.size} does not match buffer size {self.f32size}.")
        self.device.queue.write_buffer(self.buffer, 0, data.tobytes())

    def __del__(self):
        """Clean up the GPU buffer."""
        if hasattr(self, "buffer") and self.buffer is not None:
            self.buffer.destroy()

class UniformBuffer(Buffer):
    """A GPU buffer for uniform data."""
    def __init__(self, context: GPUContext, f32size: int):
        usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        super().__init__(context, f32size, usage)
    def bind_group_layout_entry(self, binding: int) -> dict[str, Any]:
        """Return a dictionary suitable for use in a bind group layout entry."""
        return super().bind_group_layout_entry(binding, wgpu.BufferBindingType.uniform)

class StorageBuffer(Buffer):
    """A GPU buffer for storage data."""
    def __init__(self, context: GPUContext, f32size: int):
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        super().__init__(context, f32size, usage)
        # allocate a staging buffer only when needed for reading back.
        self.staging_buffer = None

    def bind_group_layout_entry(self, binding: int) -> dict[str, Any]:
        """Return a dictionary suitable for use in a bind group layout entry."""
        return super().bind_group_layout_entry(binding, wgpu.BufferBindingType.storage)

    def read(self) -> np.ndarray:
        """Read data from the GPU buffer."""
        command_encoder = self.device.create_command_encoder()
        self.copy_to_staging(command_encoder)
        command_buffer = command_encoder.finish()
        self.device.queue.submit([command_buffer])
        return self.read_staging_buffer()

    def copy_to_staging(self, in_encoder: wgpu.CommandEncoder) -> None:
        """Copy data from the main buffer to the staging buffer."""
        if self.staging_buffer is None:
            self.staging_buffer = self.device.create_buffer(
                size=self.bytesize,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            )
        in_encoder.copy_buffer_to_buffer(self.buffer, 0, self.staging_buffer, 0, self.bytesize)

    def read_staging_buffer(self) -> np.ndarray:
        """Read data from the staging buffer assuming copy has happened."""
        # map the staging buffer for reading
        self.staging_buffer.map_sync(wgpu.MapMode.READ)
        # read the data
        data = np.frombuffer(self.staging_buffer.read_mapped(), dtype=np.float32).copy()
        # unmap the staging buffer
        self.staging_buffer.unmap()
        return data

  