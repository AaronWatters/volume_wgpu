from __future__ import annotations

from ..context import GPUContext
from .depth_buffer import DepthBuffer, DepthBufferGeometry, create_depth_buffer


WORKGROUP_SIZE = 64


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

        self.output_layout = self.output_depth_buffer.binding_group_layout(with_geometry=False)
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
        self.output_group = self.output_depth_buffer.binding_group(
            layout=self.output_layout,
            with_geometry=False,
        )

    def combine(self, encoder=None) -> None:
        """Copy B to the output and combine the overlapping A pixels into it."""
        my_encoder = encoder
        if my_encoder is None:
            my_encoder = self.device.create_command_encoder()
        self.depth_buffer_b.copy_buffers(self.output_depth_buffer, encoder=my_encoder)
        compute_pass = my_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.depth_buffer_a.my_group)
        compute_pass.set_bind_group(1, self.depth_buffer_b.my_group)
        compute_pass.set_bind_group(2, self.output_group)
        output_size = int(self.depth_buffer_b.geometry.width * self.depth_buffer_b.geometry.height)
        compute_pass.dispatch_workgroups((output_size + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        compute_pass.end()
        if encoder is None:
            self.device.queue.submit([my_encoder.finish()])

    def get_output(self) -> DepthBuffer:
        return self.output_depth_buffer
