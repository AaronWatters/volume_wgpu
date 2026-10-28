from __future__ import annotations

from typing import Any

import numpy as np
import wgpu

from ..context import GPUContext


WORKGROUP_SIZE = 64


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input volume must be a NumPy array.")
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array with shape (z, y, x).")
    if volume.size == 0:
        raise ValueError("Input volume must be nonempty.")
    if volume.shape[0] <= 0 or volume.shape[1] <= 0 or volume.shape[2] <= 0:
        raise ValueError("Input volume dimensions must be positive integers.")
    if volume.dtype != np.float32:
        raise TypeError("Input volume must have dtype float32.")
    return np.ascontiguousarray(volume)


class Incrementer:
    def __init__(self, shape: tuple[int, int, int], context: GPUContext | None = None):
        if len(shape) != 3:
            raise ValueError("Shape must be a 3D tuple of (z, y, x).")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            raise ValueError("Volume shape dimensions must be positive integers.")
        self.shape = tuple(int(dim) for dim in shape)
        self.context = context or GPUContext()
        self.device = self.context.device
        if self.device is None:
            raise RuntimeError("GPUContext must have an initialized device.")
        self.volume_size = int(np.prod(self.shape, dtype=np.int64))
        if self.volume_size > self.device.limits["max-buffer-size"]:
            raise ValueError("Volume is too large for this device's maximum buffer size.")

        self.shader_module = self.context.load_shader(self.device, "increment.wgsl")
        self.input_buffer = self.device.create_buffer(
            size=self.volume_size * np.dtype(np.float32).itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self.output_buffer = self.device.create_buffer(
            size=self.volume_size * np.dtype(np.float32).itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self.uniform_buffer = self.device.create_buffer(
            size=np.dtype(np.float32).itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.staging_buffer = self.device.create_buffer(
            size=self.volume_size * np.dtype(np.float32).itemsize,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.input_buffer, "offset": 0, "size": self.input_buffer.size}},
                {"binding": 1, "resource": {"buffer": self.output_buffer, "offset": 0, "size": self.output_buffer.size}},
                {"binding": 2, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_buffer.size}},
            ],
        )
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.shader_module,
                "entry_point": "main",
            },
        )

    def _validate_input(self, input_volume: np.ndarray, increment: Any) -> tuple[np.ndarray, np.float32]:
        volume = _validate_volume(input_volume)
        if volume.shape != self.shape:
            raise ValueError(f"Input volume shape {volume.shape} does not match expected shape {self.shape}.")
        if not np.isscalar(increment):
            raise TypeError("Increment must be a scalar value.")
        scalar = np.float32(increment)
        return volume, scalar

    def run(self, input_volume: np.ndarray, increment: float | np.floating = np.float32(1.0)) -> np.ndarray:
        """Run a single increment operation and return a new float32 array.

        The GPU input buffer is preserved for future processing stages. This method uploads
        the CPU buffer and reads back the output using a staging buffer, so each call includes
        a host-to-device and device-to-host transfer.
        """
        volume, scalar = self._validate_input(input_volume, increment)
        if self.volume_size <= 0:
            raise ValueError("Volume element count must be positive.")
        if self.volume_size > self.device.limits["max-buffer-size"]:
            raise ValueError("Volume is too large for the device's maximum buffer size.")

        self.device.queue.write_buffer(self.input_buffer, 0, volume.tobytes())
        self.device.queue.write_buffer(self.uniform_buffer, 0, np.asarray(scalar, dtype=np.float32).tobytes())

        encoder = self.device.create_command_encoder()
        pass_encoder = encoder.begin_compute_pass()
        pass_encoder.set_pipeline(self.pipeline)
        pass_encoder.set_bind_group(0, self.bind_group)
        pass_encoder.dispatch_workgroups((self.volume_size + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        pass_encoder.end()
        encoder.copy_buffer_to_buffer(self.output_buffer, 0, self.staging_buffer, 0, self.volume_size * np.dtype(np.float32).itemsize)
        queue = self.device.queue
        queue.submit([encoder.finish()])

        self.staging_buffer.map_sync(wgpu.MapMode.READ)
        read = self.staging_buffer.read_mapped()
        array = np.frombuffer(read, dtype=np.float32, count=self.volume_size).reshape(self.shape)
        self.staging_buffer.unmap()
        return array.copy()

    def __del__(self):
        try:
            for buffer in (self.input_buffer, self.output_buffer, self.uniform_buffer, self.staging_buffer):
                buffer.destroy()
        except Exception:
            pass
