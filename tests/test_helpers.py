from __future__ import annotations

from pathlib import Path
from typing import Any

from volume_wgpu.context import GPUContext
from volume_wgpu.gpu.increment import Incrementer


class _FakeShaderDevice:
    def __init__(self):
        self.code: str | None = None

    def create_shader_module(self, *, code: str) -> dict[str, str]:
        self.code = code
        return {"compiled": code}


def test_load_shader_concatenates_fragments_and_compiles(tmp_path: Path):
    package_path = tmp_path / "volume_wgpu"
    shaders_path = package_path / "shaders"
    shaders_path.mkdir(parents=True)
    (shaders_path / "one.wgsl").write_text("first\n", encoding="utf-8")
    (shaders_path / "two.wgsl").write_text("second\n", encoding="utf-8")
    context = GPUContext.__new__(GPUContext)
    context.package = package_path
    device = _FakeShaderDevice()

    compiled = context.load_shader(device, "one.wgsl", "two.wgsl")

    assert compiled == {"compiled": "first\nsecond\n"}
    assert device.code == "first\nsecond\n"


class _FakeBuffer:
    def __init__(self, size: int):
        self.size = size

    def destroy(self) -> None:
        return None


class _FakeQueue:
    def write_buffer(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def submit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeDevice:
    def __init__(self):
        self.queue = _FakeQueue()
        self.limits = {"max-buffer-size": 2**20}

    def create_buffer(self, *, size: int, usage: int) -> _FakeBuffer:
        _ = usage
        return _FakeBuffer(size)

    def create_bind_group_layout(self, *, entries: list[dict[str, Any]]) -> dict[str, Any]:
        return {"entries": entries}

    def create_pipeline_layout(self, *, bind_group_layouts: list[dict[str, Any]]) -> dict[str, Any]:
        return {"bind_group_layouts": bind_group_layouts}

    def create_bind_group(self, *, layout: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
        return {"layout": layout, "entries": entries}

    def create_compute_pipeline(self, *, layout: dict[str, Any], compute: dict[str, Any]) -> dict[str, Any]:
        return {"layout": layout, "compute": compute}


class _FakeContext:
    def __init__(self, device: _FakeDevice):
        self.device = device
        self.shader_module = object()
        self.calls: list[tuple[Any, tuple[str, ...]]] = []

    def load_shader(self, device: Any, *paths: str) -> object:
        self.calls.append((device, paths))
        return self.shader_module


def test_incrementer_uses_context_load_shader():
    device = _FakeDevice()
    context = _FakeContext(device)
    incrementer = Incrementer(shape=(1, 1, 1), context=context)

    assert context.calls == [(device, ("increment.wgsl",))]
    assert incrementer.shader_module is context.shader_module
    assert incrementer.pipeline["compute"]["module"] is context.shader_module
