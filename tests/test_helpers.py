from __future__ import annotations

from pathlib import Path
from typing import Any

import volume_wgpu.gpu.increment as increment_module
import volume_wgpu.helpers as helpers_module
from volume_wgpu.gpu.increment import Incrementer


class _FakeShaderDevice:
    def __init__(self):
        self.code: str | None = None

    def create_shader_module(self, *, code: str) -> dict[str, str]:
        self.code = code
        return {"compiled": code}


def test_load_shader_concatenates_fragments_and_compiles(monkeypatch, tmp_path: Path):
    package_path = tmp_path / "volume_wgpu"
    shaders_path = package_path / "shaders"
    shaders_path.mkdir(parents=True)
    (shaders_path / "one.wgsl").write_text("first\n", encoding="utf-8")
    (shaders_path / "two.wgsl").write_text("second\n", encoding="utf-8")
    monkeypatch.setattr(helpers_module.resources, "files", lambda _name: package_path)
    device = _FakeShaderDevice()

    compiled = helpers_module.load_shader(device, "one.wgsl", "two.wgsl")

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


def test_incrementer_uses_load_shader(monkeypatch):
    device = _FakeDevice()
    shader_module = object()
    calls: list[tuple[Any, tuple[str, ...]]] = []

    def fake_load_shader(dev: Any, *fragments: str) -> object:
        calls.append((dev, fragments))
        return shader_module

    monkeypatch.setattr(increment_module, "load_shader", fake_load_shader)

    incrementer = Incrementer(shape=(1, 1, 1), context=_FakeContext(device))

    assert calls == [(device, ("increment.wgsl",))]
    assert incrementer.shader_module is shader_module
    assert incrementer.pipeline["compute"]["module"] is shader_module
