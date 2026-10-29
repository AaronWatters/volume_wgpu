"""
Combine overlapping regions of two depth buffers into a new depth buffer.
This is a reference implementation in Python, not optimized for performance.
The new depth buffer should have the same geometry as the second depth buffer, 
and the second depth buffer should be transformed into the coordinate 
system of the first depth buffer.

The output depth buffer will contain the minimum depth value from 
the two input depth buffers.

The algorithm mimics the behavior of a 
compute shader that would be used in a GPU implementation.
"""

from math import ceil, floor

from volume_wgpu.reference.depth_buffer import DepthBuffer


class DepthBufferCombinerStage:
    def __init__(self, depth_buffer_a, depth_buffer_b):
        self.depth_buffer_a = depth_buffer_a
        self.depth_buffer_b = depth_buffer_b
        self.output_depth_buffer = DepthBuffer(geometry=depth_buffer_b.geometry)

    def combine(self):
        """Combine the two depth buffers into the output depth buffer."""
        # this is analogous to pipeline setup in wgpu.
        self.output_depth_buffer.data = self.depth_buffer_b.data.copy()
        self.output_depth_buffer.depth = self.depth_buffer_b.depth.copy()
        #return # debug
        length = len(self.depth_buffer_a.data)
        for offset in range(length + 3):  # shader has to handle out of bounds, so we mimic that here
            self.run_shader(offset)

    def run_shader(self, offset):
        """Simulate the shader execution for a single offset."""
        # in a real shader, this would be executed in parallel for each pixel.
        dba = self.depth_buffer_a
        dbb = self.depth_buffer_b
        location_a = dba.geometry.L_location(dba.geometry.offset_to_L(offset))
        if location_a.valid:
            # buffers a and b have different coordinate systems, so we need to transform coordinates.
            depth_a = dba.depth[offset]
            L, M, Z = location_a.L_
            x_start, y_start, _ = dba.geometry.L_to_X((L, M, Z))
            x_end = x_start + dba.geometry.dx
            y_end = y_start + dba.geometry.dy
            L_min, L_max = self._overlapping_indices(
                x_start, x_end, dbb.geometry.x0, dbb.geometry.dx
            )
            M_min, M_max = self._overlapping_indices(
                y_start, y_end, dbb.geometry.y0, dbb.geometry.dy
            )
            if L_min == L_max:
                L_max += 1
            if M_min == M_max:
                M_max += 1
            for L_b in range(L_min, L_max):
                for M_b in range(M_min, M_max):
                    location_b = dbb.geometry.L_location((L_b, M_b, Z))
                    if location_b.valid:
                        depth_b = dbb.depth[location_b.offset]
                        # combine depths
                        combined_depth = min(depth_a, depth_b)
                        self.output_depth_buffer.depth[location_b.offset] = combined_depth
                        boffset = location_b.offset
                        if depth_a < depth_b:
                            self.output_depth_buffer.data[boffset] = dba.data[offset]
                        #else:
                            #self.output_depth_buffer.data[boffset] = dbb.data[boffset]

    @staticmethod
    def _overlapping_indices(start, end, origin, step):
        """Return the half-open index range for cells intersecting an interval."""
        first = floor((start - origin) / step)
        last = ceil((end - origin) / step)
        return first, last

    def get_output(self):
        """Return the combined output depth buffer."""
        return self.output_depth_buffer
