from __future__ import annotations

import importlib.resources as resources
from typing import Any


def load_shader(device: Any, *shader_fragment_list: str) -> Any:
    package = resources.files("volume_wgpu")
    combined_shader_source = "".join(
        (package / "shaders" / shader_fragment).read_text(encoding="utf-8")
        for shader_fragment in shader_fragment_list
    )
    return device.create_shader_module(code=combined_shader_source)
