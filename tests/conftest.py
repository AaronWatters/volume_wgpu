import os

import pytest
import wgpu


def _gpu_available() -> bool:
    try:
        adapter = wgpu.gpu.request_adapter_sync()
        if adapter is None:
            return False
        device = adapter.request_device_sync()
        return device is not None
    except Exception:
        return False


@pytest.fixture(scope="session")
def gpu_context():
    strict = os.environ.get("WGPU_REQUIRE_GPU") == "1"
    if not _gpu_available():
        if strict:
            pytest.fail("WGPU_REQUIRE_GPU=1 set, but no usable wgpu adapter/device could be initialized.")
        pytest.skip("No usable wgpu adapter/device available in this environment.")
    return wgpu.gpu.request_adapter_sync().request_device_sync()


@pytest.fixture
def gpu_adapter_and_device(gpu_context):
    adapter = gpu_context.adapter
    return adapter, gpu_context
