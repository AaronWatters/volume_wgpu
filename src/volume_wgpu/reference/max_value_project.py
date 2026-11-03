"""
Maximum value projection of a volume onto a depth buffer,
reference implementation.

Initialize the depth buffer to negative infinity values and
infinity depth values.

For each voxel in the volume, project to xyz extent
and then to depth buffer.  Update the depth buffer
with value and z when value is greater than the current depth buffer entry.

In GPU shader implementation the voxels should be processed in rows
for each instance.
"""

import numpy as np
from .volume import Volume, VolumeGeometry
from .depth_buffer import DepthBuffer

class MaxProjectionStage:

    def __init__(self, from_volume: Volume, to_depth_buffer: DepthBuffer):
        self.from_volume = from_volume
        self.to_depth_buffer = to_depth_buffer

    def execute(self, init_data=-np.inf, init_depths=np.inf):
        # Initialize the depth buffer to negative infinity values and
        # infinity depth values.
        # for debugging
        self.trace = np.zeros_like(self.from_volume.source)
        self.to_depth_buffer.data[:] = init_data
        self.to_depth_buffer.depth[:] = init_depths
        # simulate shader run over rows of the volume
        vgeometry = self.from_volume.geometry
        nrows = vgeometry.nrows()
        for instance in range(nrows + 3): # simulate extra instances for GPU shader
            if (instance % 1000) == 0:
                print("processing instance", instance, "of", nrows + 3)
            self.run_shader(instance)
        # print trace value counts for debugging
        print("Trace value counts:", np.count_nonzero(self.trace), "of", self.trace.size)
        return self.to_depth_buffer

    def run_shader(self, instance: int):
        """simulate shader run for a single row instance."""
        vgeometry = self.from_volume.geometry
        dgeometry = self.to_depth_buffer.geometry
        Ni, Nj, Nk = vgeometry.shape
        i = instance // Nj
        j = instance % Nj
        if not vgeometry.valid_I([i, j, 0]):
            return
        for k in range(Nk):
            #if (k % 100) == 0:
            #    print("processing row", k, "of", Nk)
            I_ = np.array([i, j, k])
            self.trace[i, j, k] = 1
            #if not vgeometry.valid_I(I_):
            #    continue
            value = self.from_volume.data[vgeometry.I_to_offset(I_)]
            X_ = vgeometry.I_to_X(I_)
            z = X_[2]
            [min_X, max_X] = vgeometry.voxel_extent(I_)
            [(L_min, M_min), (L_max, M_max)] = dgeometry.L_extent(min_X, max_X)
            if L_min == L_max:
                L_max += 1
            if M_min == M_max:
                M_max += 1
            for L_b in range(L_min, L_max):
                for M_b in range(M_min, M_max):
                    location = dgeometry.L_location([L_b, M_b, 0])
                    if location.valid:
                        offset = location.offset
                        if value > self.to_depth_buffer.data[offset]:
                            self.to_depth_buffer.data[offset] = value
                            self.to_depth_buffer.depth[offset] = z
