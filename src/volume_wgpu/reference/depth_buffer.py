"""
A depth buffer is a pair of 2d data arrays of the same size that map to XYZ coordinates:
- a data buffer
- a depth buffer

Depth buffers are used to project XYZ values to LM image locations for display,
and for combining information from different projections.

For notational purposes
- array indices L_ = [L, M] where L is width offset and M is height offset.
- spacial coordinates X_ = [X, Y, Z] where X always maps to L and Y always maps to M and Z is a depth value.

each depth buffer has associated geometry:

- the width and height of the arrays W, H.
- the offset of position 0,0 in the array x0, y0, and
- the dx shift of one unit in x
- the dy shift of one unit in y

This is the python reference implementation.

In translation to wgpu 
- the buffers are stored as separate 2d buffers bound in a group.
- the geometry is stored in a uniform buffer bound in a separate group.

all buffers are f32 for simplicity since exact int range is sufficient.
"""

import numpy as np

class DepthGeometry:
    def __init__(self, width, height, x0, y0, dx, dy):
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
    def X_to_L(self, X_):
        """Convert XYZ coordinates to LM array indices."""
        X, Y, Z = X_
        L = int((X - self.x0) / self.dx)
        M = int((Y - self.y0) / self.dy)
        return L, M, Z
    def L_to_X(self, L_):
        """Convert LM array indices to XYZ coordinates."""
        L, M, Z = L_
        X = self.x0 + L * self.dx
        Y = self.y0 + M * self.dy
        return X, Y, Z
    def L__offset(self, L_):
        """Convert LM array indices to linear offset in the 2d array."""
        L, M, Z = L_
        return M * self.width + L
    def offset_to_L(self, offset):
        """Convert linear offset in the 2d array to LM array indices."""
        M = offset // self.width
        L = offset % self.width
        return L, M, 0
    def L_location(self, L_):
        """Return a BufferLocation for the given LM array indices."""
        L, M, Z = L_
        if self.L__valid(L_):
            offset = self.L__offset(L_)
            return BufferLocation(L_, offset=offset, valid=True)
        else:
            return BufferLocation(L_, valid=False)
    def X_location(self, X_):
        """Return a BufferLocation for the given XYZ coordinates."""
        L_ = self.X_to_L(X_)
        return self.L_location(L_)
    def X__offset(self, X_):
        """Convert XYZ coordinates to linear offset in the 2d array."""
        L_ = self.X_to_L(X_)
        return self.L__offset(L_)
    def L__valid(self, L_):
        """Check if LM array indices are within the bounds of the 2d array."""
        L, M, Z = L_
        return 0 <= L < self.width and 0 <= M < self.height
    def L_extent(self, X_1, X_2):
        """Return the LM array indices that cover the rectangle defined by two XYZ coordinates."""
        L1, M1, Z1 = self.X_to_L(X_1)
        L2, M2, Z2 = self.X_to_L(X_2)
        L_min = min(L1, L2)
        L_max = max(L1, L2)
        M_min = min(M1, M2)
        M_max = max(M1, M2)
        return (L_min, M_min), (L_max, M_max)

class BufferLocation:
    """A buffer location with data and depth and validity flag."""
    def __init__(self, L_, data=0.0, depth=float('inf'), offset=0, valid=False):
        self.L_ = L_
        self.data = data
        self.depth = depth
        self.offset = offset
        self.valid = valid

class DepthBuffer:
    def __init__(self, geometry):
        self.geometry = geometry
        #self.data = [0.0] * (geometry.width * geometry.height)
        #self.depth = [float('inf')] * (geometry.width * geometry.height)
        size = geometry.width * geometry.height
        self.data = np.zeros(size, dtype=np.float32)
        self.depth = np.full(size, float('inf'), dtype=np.float32)
    def set(self, X_, value):
        """Set the data value at the given XYZ coordinates."""
        # in wgsl this is inlined
        location = self.geometry.X_location(X_)
        if location.valid:
            self.data[location.offset] = value
        else:
            # in wgpu this is inlined and will be a no-op, but in python we raise an error
            raise ValueError(f"Coordinates {X_} are out of bounds for the depth buffer.")
    def get(self, X_):
        """Get the data value at the given XYZ coordinates."""
        # in wgsl this is inlined
        location = self.geometry.X_location(X_)
        if not location.valid:
            # in wgpu this is inlined and will be a no-op, but in python we raise an error
            raise ValueError(f"Coordinates {X_} are out of bounds for the depth buffer.")
        return self.data[location.offset]
    def set_depth(self, X_, depth):
        """Set the depth value at the given XYZ coordinates."""
        # in wgsl this is inlined
        location = self.geometry.X_location(X_)
        if not location.valid:
            # in wgpu this is inlined and will be a no-op, but in python we raise an error
            raise ValueError(f"Coordinates {X_} are out of bounds for the depth buffer.")
        self.depth[location.offset] = depth
    def get_depth(self, X_):
        """Get the depth value at the given XYZ coordinates."""
        # in wgsl this is inlined
        location = self.geometry.X_location(X_)
        if not location.valid:
            # in wgpu this is inlined and will be a no-op, but in python we raise an error
            raise ValueError(f"Coordinates {X_} are out of bounds for the depth buffer.")
        return self.depth[location.offset]
    def check_dimensions(self, data_array):
        """Check if the dimensions of the data array match the depth buffer geometry."""
        # in wgpu checking happens in python before sending to the GPU
        if len(data_array) != self.geometry.height or len(data_array[0]) != self.geometry.width:
            raise ValueError("Data array dimensions do not match depth buffer geometry.")
    def load_data0(self, data_array):
        """Load data from a 2d array into the depth buffer."""
        # loading data is done from python
        self.check_dimensions(data_array)
        for M in range(self.geometry.height):
            for L in range(self.geometry.width):
                X_ = self.geometry.L_to_X((L, M, 0))
                self.set(X_, data_array[M][L])
    def load_data(self, data_array):
        """Load data from a 2d array into the depth buffer."""
        # loading data is done from python
        self.check_dimensions(data_array)
        data_array = np.array(data_array, dtype=np.float32)
        self.data = data_array.ravel()
    def load_depth(self, depth_array):
        """Load depth values from a 2d array into the depth buffer."""
        # loading depth is done from python
        self.check_dimensions(depth_array)
        for M in range(self.geometry.height):
            for L in range(self.geometry.width):
                X_ = self.geometry.L_to_X((L, M, 0))
                self.set_depth(X_, depth_array[M][L])
    def get_numpy_arrays(self):
        """Return the data and depth buffers as 2d numpy arrays."""
        import numpy as np
        data_array = np.array(self.data).reshape((self.geometry.height, self.geometry.width))
        depth_array = np.array(self.depth).reshape((self.geometry.height, self.geometry.width))
        return data_array, depth_array
    