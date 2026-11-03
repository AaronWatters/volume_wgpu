
import numpy as np
from volume_wgpu.reference.depth_buffer import DepthBuffer, DepthGeometry
from volume_wgpu.reference.volume import Volume, VolumeGeometry
from volume_wgpu.reference.max_value_project import MaxProjectionStage

# read in "vis_male.npz"
data = np.load("vis_male.npz")

print ("read in vis_male.npz", data.keys())
arr = data["arr"]
print("array shape", arr.shape)

V = Volume(arr)
(width, height, depth) = arr.shape
Dgeometry = DepthGeometry(width, height, 0, 0, 1, 1)
Dbuffer = DepthBuffer(Dgeometry)

M = MaxProjectionStage(V, Dbuffer)
M.execute(
    init_data=0,
    init_depths=np.inf
)

print("Max projection executed.")
#out = Dbuffer.get_data_numpy_array()
out, _ = Dbuffer.get_numpy_arrays()
print("Output shape", out.shape)
print("output extrema", out.min(), out.max())

# save the output as an image using PIL
filename = "ref_head_output.png"
scaled_out = (out - out.min()) / (out.max() - out.min()) * 255
from PIL import Image
image = Image.fromarray(scaled_out.astype(np.uint8))
image.save(filename)
print("Saved output image to", filename)
