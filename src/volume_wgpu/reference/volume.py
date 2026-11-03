"""
A volume is a three-dimensional array of data
with an associated geometry that defines its spatial extent and resolution.

The geometry is defined by a 4x4 affine matrix M usually representing
a rotation, scaling, and translation.  For convenience the inverse
transformation matrix Minv is also stored.

For notational purposes:
- X_ = [x, y, z] is spacial "model" coordinates.
- I_ = [i, j, k] is the index into the volume array.
- X_1 = [x, y, z, 1] -- for homogeneous coordinates.
- I_1 = [i, j, k, 1] -- for homogeneous coordinates.

Transformations:
- I_1 = Minv * X_1
- X_1 = M * I_1

On the GPU the volume is stored as a linear f32 array
where an element at index I_ = [i, j, k] is stored at linear index i + j * Ni + k * Ni * Nj.
"""

import numpy as np

class VolumeGeometry:

    def __init__(self, M: np.ndarray, shape: np.ndarray):
        self.set_M(M)
        self.shape = shape

    def nrows(self):
        Ni, Nj, Nk = self.shape
        return Ni * Nj

    def row_size(self):
        Ni, Nj, Nk = self.shape
        return Nk

    def set_M(self, M: np.ndarray):
        self.M = M
        self.Minv = np.linalg.inv(M)

    def I_to_X(self, I_: np.ndarray) -> np.ndarray:
        I_1 = np.append(I_, 1)
        X_1 = self.M @ I_1
        return X_1[:3]

    def X_to_I(self, X_: np.ndarray) -> np.ndarray:
        X_1 = np.append(X_, 1)
        I_1 = self.Minv @ X_1
        return I_1[:3]

    def I_to_offset(self, I_: np.ndarray):
        Ni, Nj, Nk = self.shape
        i, j, k = I_
        index = (i * Nj + j) * Nk + k
        return index

    def offset_to_I(self, offset: int) -> np.ndarray:
        Ni, Nj, Nk = self.shape
        k = offset % Nk
        j = (offset // Nk) % Nj
        i = offset // (Nj * Nk)
        return np.array([i, j, k])

    def X_to_offset(self, X_: np.ndarray) -> int:
        I_ = self.X_to_I(X_)
        return self.I_to_offset(I_)

    def offset_to_X(self, offset: int) -> np.ndarray:
        I_ = self.offset_to_I(offset)
        return self.I_to_X(I_)

    def valid_I(self, I_: np.ndarray) -> bool:
        Ni, Nj, Nk = self.shape
        i, j, k = I_
        return 0 <= i < Ni and 0 <= j < Nj and 0 <= k < Nk

    def voxel_extent(self, I_: np.ndarray) -> np.ndarray:
        """return max and min xyz coordinates of the voxel at index I_"""
        i, j, k = I_
        [min_x, min_y, min_z] = [max_x, max_y, max_z] = self.I_to_X(np.array([i, j, k]))
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    (x,y,z) = self.I_to_X(np.array([i + di, j + dj, k + dk]))
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    min_z = min(min_z, z)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    max_z = max(max_z, z)
        return np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]])

class VolumeLocation:
    def __init__(self, I_: np.ndarray, geometry: VolumeGeometry):
        self._I = I_
        self.valid = geometry.valid_I(I_)
        self.offset = geometry.I_to_offset(I_)
        self.X_ = geometry.I_to_X(_I)

class Volume:
    def __init__(self, data: np.ndarray, geometry: VolumeGeometry=None):
        if geometry is None:
            geometry = VolumeGeometry(np.eye(4), data.shape)
        self.geometry = geometry
        self.load_and_validate_data(data)

    def load_and_validate_data(self, data):
        # check that the data matches the expected shape
        if data.shape != self.geometry.shape:
            raise ValueError("Data shape does not match geometry shape")
        self.data = data.astype(np.float32).ravel()
        # for debugging
        self.source = data
