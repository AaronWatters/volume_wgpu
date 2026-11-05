
"""
Max value projection GPU implementation.
"""

import numpy as np

WORKGROUP_SIZE = 64

class MaxValueProjectStage:

    def __init__(self, from_volume, to_depth_buffer, context):
        self.from_volume = from_volume
        self.to_depth_buffer = to_depth_buffer
        self.context = context
        device = context.device
        volume_layout = from_volume.binding_group_layout()
        depth_layout = to_depth_buffer.binding_group_layout()
        self.pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[volume_layout, depth_layout]
        )
        self.shader_module = self.context.load_shader(
            "depth_buffer.wgsl",
            "volume.wgsl",
            "max_value.wgsl"
        )
        self.pipeline = device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": "main"},
        )
        self.volume_group = from_volume.binding_group(layout=volume_layout)
        self.depth_group = to_depth_buffer.binding_group(layout=depth_layout)

    def execute(self, encoder=None):
        my_encoder = encoder
        device = self.context.device
        if my_encoder is None:
            my_encoder = device.create_command_encoder()
        nrows = self.from_volume.geometry.nrows()
        my_pass = my_encoder.begin_compute_pass()
        my_pass.set_pipeline(self.pipeline)
        my_pass.set_bind_group(0, self.volume_group)
        my_pass.set_bind_group(1, self.depth_group)
        my_pass.dispatch_workgroups((nrows + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        my_pass.end()
        if encoder is None:
            device.queue.submit([my_encoder.finish()])
    