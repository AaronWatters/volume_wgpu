from __future__ import annotations

import numpy as np

from ..context import GPUContext
from .depth_buffer import DepthBuffer, DepthBufferGeometry


WORKGROUP_SIZE = 64


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


class DepthBufferCombinerStage:
    """Combine overlapping GPU depth buffers into a buffer with B's geometry."""

    def __init__(
        self,
        depth_buffer_a: DepthBuffer,
        depth_buffer_b: DepthBuffer,
        context: GPUContext | None = None,
    ):
        self.depth_buffer_a = depth_buffer_a
        self.depth_buffer_b = depth_buffer_b
        self.context = context or depth_buffer_a.context
        if depth_buffer_b.context is not self.context:
            raise ValueError("Both depth buffers must use the same GPU context.")
        self.device = self.context.device
        self.output_depth_buffer = DepthBuffer()
        self.output_depth_buffer.init(self.context, depth_buffer_b.geometry)
        self.shader_module = self.context.load_shader("depth_buffer.wgsl", "combine_overlap.wgsl")

        self.output_layout = self.device.create_bind_group_layout(
            entries=[
                self.output_depth_buffer.data_buffer.bind_group_layout_entry(0),
                self.output_depth_buffer.depth_buffer.bind_group_layout_entry(1),
            ]
        )
        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[
                self.depth_buffer_a.my_layout,
                self.depth_buffer_b.my_layout,
                self.output_layout,
            ]
        )
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": "main"},
        )
        self.output_group = self.device.create_bind_group(
            layout=self.output_layout,
            entries=[
                self.output_depth_buffer.data_buffer.binding_group_entry(0),
                self.output_depth_buffer.depth_buffer.binding_group_entry(1),
            ],
        )

    def combine(self) -> None:
        """Copy B to the output and combine the overlapping A pixels into it."""
        encoder = self.device.create_command_encoder()
        size = self.depth_buffer_b.data_buffer.bytesize
        encoder.copy_buffer_to_buffer(
            self.depth_buffer_b.data_buffer.buffer,
            0,
            self.output_depth_buffer.data_buffer.buffer,
            0,
            size,
        )
        encoder.copy_buffer_to_buffer(
            self.depth_buffer_b.depth_buffer.buffer,
            0,
            self.output_depth_buffer.depth_buffer.buffer,
            0,
            size,
        )
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.depth_buffer_a.my_group)
        compute_pass.set_bind_group(1, self.depth_buffer_b.my_group)
        compute_pass.set_bind_group(2, self.output_group)
        output_size = int(self.depth_buffer_b.geometry.width * self.depth_buffer_b.geometry.height)
        compute_pass.dispatch_workgroups((output_size + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

    def get_output(self) -> DepthBuffer:
        return self.output_depth_buffer
