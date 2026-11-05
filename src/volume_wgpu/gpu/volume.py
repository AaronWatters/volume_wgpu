

"""
Volume GPU implementation.
"""

import numpy as np
from ..context import GPUContext
from ..buffer import StorageBuffer, UniformBuffer

class VolumeGeometry:
    """
    Wrapper for gpu resident volume geometry stored in a uniform buffer.
    """

    # 2 4x4 matrices for transform and inverse, plus 3 floats for shape + 1 float padding for alignment
    f32size = (2 * 4 * 4) + 3 + 1

    def __init__(self, context, shape):
        self.context = context
        self.buffer = UniformBuffer(context, f32size=self.f32size)  # 32 floats for transform and inverse
        self.shape = np.array(shape, dtype=np.float32)
        self.transform = np.eye(4, dtype=np.float32)
        self.inverse = np.eye(4, dtype=np.float32)

    def set_transform(self, transform):
        self.transform = transform
        self.inverse = np.linalg.inv(transform).astype(np.float32)

    def set_inverse(self, inverse):
        self.inverse = inverse
        self.transform = np.linalg.inv(inverse).astype(np.float32)

    def get_numpy_array(self):
        """
        Returns the volume geometry as a numpy array.
        """
        # use column major order for the matrices for wgsl compatibility
        return np.concatenate([
            self.transform.flatten(order='F'), 
            self.inverse.flatten(order='F'),
            self.shape.flatten(),
            np.array([0], dtype=np.float32)  # padding for alignment
            ])

    def buffer_f32size(self):
        [I, J, K] = self.shape
        return I * J * K

    # Shaders for volumes often iterate across rows.
    def nrows(self):
        [I, J, K] = self.shape
        return I * J # iterate across K dimension

    def write(self):
        """
        Writes the volume geometry to the GPU buffer.
        """
        self.buffer.write(self.get_numpy_array())

    def bind_group_layout_entry(self, binding):
        return self.buffer.bind_group_layout_entry(binding)

    def binding_group_entry(self, binding):
        return self.buffer.binding_group_entry(binding)

class Volume:
    """
    Wrapper for gpu resident volume data stored in a storage buffer.
    with associated geometry stored in a uniform buffer.
    """

    def __init__(self, context, geometry):
        self.context = context
        self.geometry = geometry
        self.f32size = geometry.buffer_f32size()
        self.data_buffer = StorageBuffer(context, f32size=self.f32size)

    def set_transform(self, transform):
        self.geometry.set_transform(transform)
        self.geometry.write()

    def binding_group_layout(self, with_geometry=True):
        entries = [self.data_buffer.bind_group_layout_entry(0)]
        if with_geometry:
            entries.append(self.geometry.bind_group_layout_entry(1))
        device = self.context.device
        return device.create_bind_group_layout(entries=entries)

    def binding_group(self, layout=None, with_geometry=True):
        if layout is None:
            layout = self.binding_group_layout(with_geometry=with_geometry)
        entries = [self.data_buffer.binding_group_entry(0)]
        if with_geometry:
            entries.append(self.geometry.binding_group_entry(1))
        device = self.context.device
        return device.create_bind_group(layout=layout, entries=entries)

    def write_data(self, data):
        """
        Writes the volume data to the GPU buffer.
        """
        self.data_buffer.write(data)
        self.geometry.write()

    def get_data_numpy_array(self, reshape=True):
        """
        Returns the volume data as a numpy array.
        """
        data = self.data_buffer.read()
        if reshape:
            data = data.reshape(self.geometry.shape)
        return data

def create_volume(context, data, geometry=None):
    """
    Creates and initializes a GPU volume from a numpy array and optional geometry.
    """
    if geometry is None:
        geometry = VolumeGeometry(context, data.shape)
    geometry.write()
    volume = Volume(context, geometry)
    volume.write_data(data.ravel())
    return volume
