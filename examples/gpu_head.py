
import numpy as np
from volume_wgpu.gpu.max_value_project import MaxValueProjectStage
from volume_wgpu.context import GPUContext
from volume_wgpu.gpu.depth_buffer import create_depth_buffer
from volume_wgpu.gpu.depth_buffer import DepthBufferGeometry
from volume_wgpu.gpu.volume import Volume, VolumeGeometry


# read in "vis_male.npz"
data = np.load("vis_male.npz")

context = GPUContext()

print ("read in vis_male.npz", data.keys())
arr = data["arr"]
print("array shape", arr.shape)
print("array extrema", arr.min(), arr.max())

(width, height, depth) = arr.shape
depth_buffer_geometry = DepthBufferGeometry(context, width=width, height=height)
data = np.full((height, width), 2, dtype=np.float32)
depth = np.full((height, width), np.inf, dtype=np.float32)
depth_buffer = create_depth_buffer(context, depth_buffer_geometry, data=data, depth=depth)

geometry = VolumeGeometry(context, arr.shape)
volume = Volume(geometry=geometry, context=context)
volume.write_data(arr)

stage = MaxValueProjectStage(volume, depth_buffer, context)
stage.execute()

values = depth_buffer.get_data_numpy_array()
print ("output shape", values.shape)
print ("output extrema", values.min(), values.max())
print("first few values", values.flatten()[:30])

filename = "gpu_head_output.png"
out = values
scaled_out = (out - out.min()) / (out.max() - out.min()) * 255

from PIL import Image
image = Image.fromarray(scaled_out.astype(np.uint8))
image.save(filename)
print("Saved output image to", filename)