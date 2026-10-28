from __future__ import annotations

import importlib.resources as resources
import os
from dataclasses import dataclass
from typing import Any

import wgpu


class GPUUnavailableError(RuntimeError):
    """Raised when a usable wgpu adapter/device cannot be initialized."""


@dataclass
class GPUContext:
    """Small reusable wgpu adapter/device context."""

    adapter: Any | None = None
    device: Any | None = None
    package: Any | None = None

    def __post_init__(self) -> None:
        if self.adapter is None:
            self.adapter = wgpu.gpu.request_adapter_sync()
        if self.adapter is None:
            reason = "No wgpu adapter is available in this environment."
            if os.environ.get("WGPU_REQUIRE_GPU") == "1":
                raise GPUUnavailableError(reason)
            raise GPUUnavailableError(reason)
        if self.device is None:
            self.device = self.adapter.request_device_sync()
        if self.device is None:
            reason = "wgpu adapter was found, but device creation failed."
            if os.environ.get("WGPU_REQUIRE_GPU") == "1":
                raise GPUUnavailableError(reason)
            raise GPUUnavailableError(reason)
        if self.package is None:
            self.package = resources.files("volume_wgpu")

    def load_fragment(self, path: str) -> str:
        return (self.package / "shaders" / path).read_text(encoding="utf-8")

    def load_shader(self, device: Any, *paths: str) -> Any:
        combined_shader_source = "".join(self.load_fragment(path) for path in paths)
        return device.create_shader_module(code=combined_shader_source)

    @property
    def adapter_info(self) -> dict[str, Any]:
        info = getattr(self.adapter, "info", {})
        if hasattr(info, "__iter__"):
            try:
                items = dict(info)
                if items:
                    return items
            except Exception:
                pass
        return {
            "backend_type": getattr(info, "backend_type", None),
            "device": getattr(info, "device", None),
            "description": getattr(info, "description", None),
        }

    def __repr__(self) -> str:
        info = self.adapter_info
        name = info.get("device") or info.get("description") or "unknown"
        backend = info.get("backend_type") or "unknown"
        return f"GPUContext(adapter={name}, backend={backend})"
