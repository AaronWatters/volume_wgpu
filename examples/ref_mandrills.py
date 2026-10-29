from pathlib import Path

import numpy as np
from PIL import Image

from volume_wgpu.reference.combine_overlap import DepthBufferCombinerStage
from volume_wgpu.reference.depth_buffer import DepthBuffer, DepthGeometry

"""
with Image.open("image.png") as image:
    red = np.array(image.convert("RGB").getchannel("R"))
"""

def main() -> None:
    examples_dir = Path(__file__).parent
    # read just the red channel of the mandrill image for simplicity
    image = np.array(Image.open(examples_dir / "mandrill.png").convert("RGB").getchannel("R"))
    #image = image[:, :, 0]  # Select the red channel
    height, width = image.shape
    buffer_a_geometry = DepthGeometry(
        width=width,
        height=height,
        x0=0.0,
        y0=0.0,
        dx=1.0 / width,
        dy=1.0 / height,
    )
    buffer_b_geometry = DepthGeometry(
        width=width,
        height=height,
        x0=-2.0,
        y0=-2.0,
        dx=4.0 / width,
        dy=4.0 / height,
    )

    buffer_a = DepthBuffer(buffer_a_geometry)
    #buffer_a.load_data(image.tolist())
    buffer_a.load_data(image)
    buffer_a.load_depth(np.full((height, width), 2.0, dtype=np.float32).tolist())

    b_depth = 3.0
    #b_depth = 1.0 # debug
    buffer_b = DepthBuffer(buffer_b_geometry)
    buffer_b.load_data(image)
    buffer_b.load_depth(np.full((height, width), 3.0, dtype=np.float32).tolist())

    combiner = DepthBufferCombinerStage(buffer_a, buffer_b)
    combiner.combine()
    values, _ = combiner.get_output().get_numpy_arrays()
    #values, _ = buffer_a.get_numpy_arrays() # debug
    #values = image # debug
    #values = np.array(image.tolist(), dtype=np.float32).reshape((height, width))  # debug
    #values = np.array(combiner.depth_buffer_b.data, dtype=np.float32).reshape((height, width))  # debug
    output = np.clip(values, 0.0, 255.0).astype(np.uint8)
    # save the output image as black and white for simplicity
    Image.fromarray(output).save(examples_dir / "out_mandrills_py.png")


if __name__ == "__main__":
    main()
