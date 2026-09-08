# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from anima_master_adapter import AnimaMasterAdapter, find_anima_master_plugin, reference_capacity
from generation_config import build_route_registry
from generation_contracts import CharacterIdentitySpecV1, GenerationSpecV1, ReferenceBindingV1, SceneContextV1, WardrobeSpecV1
from generation_engine import GenerationEngine
from generation_profiles import default_model_profile_registry


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class Plugin:
    def __init__(self, root):
        self._runtime = SimpleNamespace(root=root)
        self.config = {"img2img_enabled": False, "send_result_to_chat": True, "cfg": 3}
        self.output = root / "generated.png"
        self.output.write_bytes(PNG)
        self.payload = {"ok": True, "outputs": [str(self.output)], "task_id": "task-1"}
        self.calls = []
        self.ready = {"ok": True}
        self.allowed = True

    async def _generate_payload(self, event, prompt, *, prepared_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "prepared_prompt": prepared_prompt, **kwargs})
        return self.payload

    async def _generate(self, *args, **kwargs):
        raise AssertionError("chat delivery must not be called")

    def _is_allowed(self, event):
        return self.allowed

    async def _ensure_comfyui_ready(self, event):
        return self.ready

    async def _run_tool(self, args):
        source = Path(args[args.index("--input") + 1])
        self.calls.append({"args": args, "source": source, "bytes": source.read_bytes()})
        return self.payload


def adapter(plugin):
    return AnimaMasterAdapter(plugin, lambda: SimpleNamespace(send=AsyncMock()), AsyncMock(return_value="archive.png"))


def test_discovery_uses_live_metadata_and_skips_disabled_instances(tmp_path):
    plugin = Plugin(tmp_path)
    for registered in (plugin, SimpleNamespace(star_cls=plugin, activated=True)):
        context = SimpleNamespace(get_registered_star=lambda name: registered)
        assert find_anima_master_plugin(context) is plugin
    context = SimpleNamespace(get_registered_star=lambda name: SimpleNamespace(star_cls=plugin, activated=False))
    assert find_anima_master_plugin(context) is None
    assert find_anima_master_plugin(None) is None


@pytest.mark.asyncio
async def test_generation_preserves_prompt_negative_size_and_shared_configuration(tmp_path):
    plugin = Plugin(tmp_path)
    original = deepcopy(plugin.config)
    bridge = adapter(plugin)
    result = await bridge.execute("1girl, summer dress, city street", negative_prompt="winter coat", image_size="832x1216")
    assert result.ok and result.image_path == "archive.png"
    assert result.task_id == "task-1" and result.generation_completed
    assert plugin.calls == [{
        "prompt": "1girl, summer dress, city street",
        "prepared_prompt": "1girl, summer dress, city street",
        "prepared_prompt_summary": {"source": "image_companion", "operation": "text2img"},
        "negative_prompt": "winter coat", "width": 832, "height": 1216,
    }]
    assert plugin.config == original
    bridge.materialize.assert_awaited_once_with(str(plugin.output))


@pytest.mark.asyncio
async def test_failure_does_not_read_stale_outputs_or_retry(tmp_path):
    plugin = Plugin(tmp_path)
    plugin.payload["ok"] = False
    plugin.payload["error"] = "not_permitted"
    bridge = adapter(plugin)
    result = await bridge.execute("test")
    assert not result.ok and "not_permitted" in result.note
    assert len(plugin.calls) == 1
    bridge.materialize.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing_outputs", "missing_file", "archive_error"])
async def test_completed_generation_retains_materialization_failure_state(tmp_path, mode):
    plugin = Plugin(tmp_path)
    bridge = adapter(plugin)
    if mode == "missing_outputs":
        plugin.payload["outputs"] = []
    elif mode == "missing_file":
        bridge.materialize.return_value = ""
    else:
        bridge.materialize.side_effect = OSError("disk full")
    result = await bridge.execute("test")
    assert result.generation_completed
    assert result.error_code == "result_materialization_failed"
    assert result.failure_stage == "result_materialization"
    assert not result.image_path


@pytest.mark.asyncio
async def test_reference_opt_in_and_explicit_workspace_input(tmp_path):
    plugin = Plugin(tmp_path)
    ref = ReferenceBindingV1(reference_id="current", path=str(plugin.output), roles=("edit_source",))
    bridge = adapter(plugin)
    result = await bridge.execute("change dress", references=(ref,), operation="edit")
    assert not result.ok and "img2img_enabled" in result.note
    assert not plugin.calls
    plugin.config["img2img_enabled"] = True
    result = await bridge.execute("change dress", references=(ref,), operation="edit", negative_prompt="coat", image_size="1024x1024")
    assert result.ok and result.submitted_reference_ids == ("current",)
    assert "negative_prompt:upstream_edit_defaults" in result.degraded_capabilities
    assert "size:upstream_reference_dimensions" in result.degraded_capabilities
    call = plugin.calls[0]
    assert call["bytes"] == PNG
    assert call["source"].is_relative_to(tmp_path / "workspace" / "companion_references")
    assert not call["source"].exists()
    assert call["args"][0] == "edit"
    assert "latest" not in call["args"]
    assert plugin.output.exists()


@pytest.mark.asyncio
async def test_reference_permissions_and_readiness_are_preserved(tmp_path):
    plugin = Plugin(tmp_path)
    plugin.config["img2img_enabled"] = True
    ref = ReferenceBindingV1(reference_id="current", path=str(plugin.output))
    plugin.allowed = False
    result = await adapter(plugin).execute("test", references=(ref,))
    assert not result.ok and not plugin.calls
    plugin.allowed = True
    plugin.ready = {"ok": False, "error": "comfyui_offline"}
    result = await adapter(plugin).execute("test", references=(ref,))
    assert "comfyui_offline" in result.note and not plugin.calls


@pytest.mark.asyncio
async def test_incompatible_version_and_invalid_size_never_submit(tmp_path):
    plugin = Plugin(tmp_path)
    assert not (await adapter(plugin).execute("test", image_size="2K")).ok
    assert not plugin.calls

    async def old_generator(event, prompt):
        raise AssertionError("old generator must not be invoked")

    plugin._generate_payload = old_generator
    result = await adapter(plugin).execute("test")
    assert result.error_code == "route_unavailable"
    assert reference_capacity(None) == 0


@pytest.mark.asyncio
async def test_unified_route_runs_and_shadow_does_not_submit(tmp_path):
    plugin = Plugin(tmp_path)
    registry, validation = build_route_registry({"engine": {"routes": [{
        "name": "anima-local", "backend": "anima_master", "model_profile": "anima", "operation": "text2img",
    }]}})
    assert validation.ok
    engine = GenerationEngine(default_model_profile_registry(), registry, {"anima_master": adapter(plugin)})
    spec = GenerationSpecV1(
        schema_version=1, request_id="one", operation="text2img", user_request="summer dress",
        scene=SceneContextV1(), character=CharacterIdentitySpecV1(), wardrobe=WardrobeSpecV1(),
    )
    result = await engine.generate(spec, "anima-local")
    assert result.ok and result.backend == "anima_master"
    assert "summer dress" in plugin.calls[0]["prepared_prompt"]
    engine.shadow_mode = True
    result = await engine.generate(spec, "anima-local")
    assert "shadow_mode" in result.note
    assert len(plugin.calls) == 1


def test_route_rejects_ignored_workflow_override():
    _, validation = build_route_registry({"engine": {"routes": [{
        "name": "anima-local", "backend": "anima_master", "workflow": "other.json",
    }]}})
    assert not validation.ok


@pytest.mark.asyncio
async def test_adapter_propagates_cancellation(tmp_path):
    plugin = Plugin(tmp_path)

    async def cancel(event, prompt, *, prepared_prompt=None, **kwargs):
        raise asyncio.CancelledError()

    plugin._generate_payload = cancel
    with pytest.raises(asyncio.CancelledError):
        await adapter(plugin).execute("test")


@pytest.mark.asyncio
async def test_native_runtime_routes_to_anima_and_preserves_other_backends(tmp_path):
    from test_migrated_photo_prompt_reliability import _PhotoReliabilityHarness

    runtime = _PhotoReliabilityHarness(tmp_path)
    runtime.photo_generation_backend = "anima_master"
    plugin = Plugin(tmp_path)
    bridge = adapter(plugin)
    bridge.materialize.return_value = str(plugin.output)
    runtime._create_anima_master_adapter = lambda session_key: bridge
    runtime._local_photo_generation_busy_state = lambda **kwargs: None
    backend, path, note = await runtime._generate_photo_image(
        prompt_text="a landscape, blue sky", workflow_kind="text2img", session_key="test-session",
    )
    assert backend == "Anima 绘图大师" and path == str(plugin.output)
    assert note == "ok"
    assert plugin.calls and not runtime.external_calls


@pytest.mark.asyncio
async def test_runtime_adapter_materializes_real_image_and_handles_star_metadata(tmp_path):
    from astrbot_plugin_image_companion.image_runtime import ImageGenerationRuntime

    plugin = Plugin(tmp_path)
    runtime = ImageGenerationRuntime.__new__(ImageGenerationRuntime)
    runtime.context = SimpleNamespace(get_registered_star=lambda name: SimpleNamespace(star_cls=plugin))
    runtime._parse_message_session = lambda key: None
    runtime._save_external_generated_image = AsyncMock(return_value=str(tmp_path / "archive.png"))
    result = await runtime._create_anima_master_adapter("private:test").execute("test")
    assert result.ok
    runtime._save_external_generated_image.assert_awaited_once_with(PNG, session_key="private:test", ext=".png")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANIMA_MASTER_SOURCE"), reason="optional upstream checkout is unavailable")
async def test_upstream_prepared_payload_contract(tmp_path):
    source = Path(os.environ["ANIMA_MASTER_SOURCE"]) / "generation_task.py"
    spec = importlib.util.spec_from_file_location("anima_upstream_generation_task", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = Plugin(tmp_path)
    run_tool = AsyncMock(return_value=plugin.payload)
    optimizer = AsyncMock(side_effect=AssertionError("prepared prompts must not be rewritten"))
    recorder = SimpleNamespace(
        build_generation_start=lambda **kwargs: {"task_id": "upstream-task"},
        mark_prompt_built=lambda *args: None,
        mark_completed=lambda *args, **kwargs: None,
        write=lambda task: None,
    )
    task = module.GenerationTaskRunner(
        task_recorder=recorder, image_inputs=SimpleNamespace(last_summary={}),
        reference_context=SimpleNamespace(last_summary={}), is_allowed=lambda event: True,
        ensure_ready=AsyncMock(return_value={"ok": True}), wants_reference_image=lambda prompt: False,
        augment_reference_image=optimizer, augment_quoted_spell=lambda event, prompt: prompt,
        build_prompt=optimizer, prompt_summary=lambda: {}, run_tool=run_tool,
        get_bool=lambda key, default: default, get_int=lambda key, default: default,
        get_float=lambda key, default: default, get_str=lambda key, default: default,
        shorten=lambda text, limit=1800: text[:limit],
    )
    plugin._generate_payload = task.generate_payload
    result = await adapter(plugin).execute("summer dress", negative_prompt="winter coat", image_size="832x1216")
    assert result.ok and result.task_id == "upstream-task"
    optimizer.assert_not_awaited()
    run_tool.assert_awaited_once_with([
        "generate", "--prompt", "summer dress", "--width", "832", "--height", "1216",
        "--override-size", "--negative-prompt", "winter coat",
    ])
