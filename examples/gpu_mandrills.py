from pathlib import Path

import numpy as np
from PIL import Image

from volume_wgpu import GPUContext
from volume_wgpu.gpu.combine_overlap import DepthBufferCombinerStage, create_depth_buffer
from volume_wgpu.gpu.depth_buffer import DepthBufferGeometry


def main() -> None:
    examples_dir = Path(__file__).parent
    image = np.array(Image.open(examples_dir / "mandrill.png").convert("L"), dtype=np.float32)
    height, width = image.shape
    context = GPUContext()
    geometry_a = DepthBufferGeometry(context, width, height, dx=1.0 / width, dy=1.0 / height)
    geometry_b = DepthBufferGeometry(
        context, width, height, x0=-2.0, y0=-2.0, dx=4.0 / width, dy=4.0 / height
    )
    buffer_a = create_depth_buffer(
        context,
        geometry_a,
        image,
        np.full_like(image, 2.0),
    )
    buffer_b = create_depth_buffer(
        context,
        geometry_b,
        image,
        np.full_like(image, 3.0),
    )
    combiner = DepthBufferCombinerStage(buffer_a, buffer_b, context=context)
    combiner.combine()
    values = combiner.get_output().get_data_numpy_array()
    Image.fromarray(np.clip(values, 0.0, 255.0).astype(np.uint8)).save(
        examples_dir / "out_mandrills_gpu.png"
    )


if __name__ == "__main__":
    main()
