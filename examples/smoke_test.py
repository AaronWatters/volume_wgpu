from __future__ import annotations

import numpy as np

from volume_wgpu import GPUContext, Incrementer
from volume_wgpu.reference.increment import increment_reference


def main() -> None:
    context = GPUContext()
    adapter_info = context.adapter_info
    volume = np.arange(7 * 13 * 19, dtype=np.float32).reshape(7, 13, 19)
    operation = Incrementer(shape=volume.shape, context=context)
    result = operation.run(volume, increment=np.float32(1.25))
    expected = increment_reference(volume, np.float32(1.25))
    if not np.allclose(result, expected, rtol=1e-6, atol=1e-6):
        raise AssertionError("GPU increment result did not match the NumPy reference.")
    print(f"Adapter: {adapter_info.get('device') or 'unknown'}")
    print(f"Backend: {adapter_info.get('backend_type') or 'unknown'}")
    print(f"Volume shape: {volume.shape}, result sample: {result[0, 0, :5]}")
    print("GPU smoke test passed.")


if __name__ == "__main__":
    main()
