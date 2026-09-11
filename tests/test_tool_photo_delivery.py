from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_image_companion import image_runtime as module
from test_migrated_photo_prompt_reliability import _PhotoReliabilityHarness

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class Image:
    def __init__(self, file):
        self.file = file


class Event:
    def __init__(self, umo="qq:FriendMessage:user"):
        self.unified_msg_origin = umo
        self.send = AsyncMock()

    def get_sender_id(self):
        return self.unified_msg_origin.rsplit(":", 1)[-1]

    def chain_result(self, chain):
        return SimpleNamespace(chain=chain)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "Image", Image)
    obj = module.ImageGenerationRuntime.__new__(module.ImageGenerationRuntime)
    obj._image_service = SimpleNamespace(image_setting=lambda _name, default: default)
    obj._image_owner = SimpleNamespace()
    obj.context = SimpleNamespace(send_message=AsyncMock())
    obj.custom_photo_tool_name = "comfyui_generate"
    async def save(raw, *, session_key, ext):
        target = tmp_path / (session_key.rsplit(":", 1)[-1] + ext)
        target.write_bytes(raw)
        return str(target)
    obj._save_external_generated_image = AsyncMock(side_effect=save)
    return obj


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    r"E:\AI\AstrBotLauncher-0.3.0\AstrBot\workspace\outputs\images\20260910-003503-example_comfyui_1.png",
    r"C:\我的 图片\人物 (白衣)\自拍.PNG",
    r"\\nas\images\朋友 合照.webp",
    "C:/Program Files/AstrBot/image output.jpg",
    "/tmp/中文 图片/selfie.png",
    "file:///C:/image%20output/selfie.png",
])
async def test_extracts_complete_paths_from_tool_prose(runtime, path):
    candidates = []
    async def resolve(value, **_):
        candidates.append(value)
        return ("/archive.png", "ok") if value == path else ("", "missing")
    runtime._resolve_custom_tool_image_candidate = resolve
    result = await runtime._parse_custom_tool_photo_result(f'ComfyUI 已生成并发送图片："{path}"。', session_key="test")
    assert result[0] == "/archive.png"
    assert path in candidates
    assert "E:\\" not in candidates


@pytest.mark.asyncio
async def test_extracts_relative_paths_with_spaces_from_tool_prose(runtime):
    candidates = []

    async def resolve(value, **_):
        candidates.append(value)
        return ("/archive.png", "ok") if value == "outputs/final image.png" else ("", "missing")

    runtime._resolve_custom_tool_image_candidate = resolve
    result = await runtime._parse_custom_tool_photo_result(
        "ComfyUI 输出位于 outputs/final image.png。",
        session_key="test",
    )
    assert result[0] == "/archive.png"
    assert "outputs/final image.png" in candidates


@pytest.mark.asyncio
async def test_image_file_uri_is_decoded_and_archived(runtime, tmp_path):
    source = tmp_path / "带空格 的图片.png"
    source.write_bytes(PNG)
    path, note = await runtime._resolve_custom_tool_image_candidate(source.as_uri(), session_key="qq:FriendMessage:user")
    assert Path(path).read_bytes() == PNG and "归档" in note
    assert path != str(source)


@pytest.mark.asyncio
async def test_small_data_uri_is_validated_and_archived(runtime):
    encoded = base64.b64encode(PNG).decode("ascii")
    path, note = await runtime._resolve_custom_tool_image_candidate(
        f"data:image/png;base64,{encoded}",
        session_key="test",
    )
    assert path and Path(path).read_bytes() == PNG
    assert "base64" in note


@pytest.mark.asyncio
async def test_small_data_uri_is_extracted_from_tool_prose(runtime):
    encoded = base64.b64encode(PNG).decode("ascii")
    path, note = await runtime._parse_custom_tool_photo_result(
        f"工具返回图片：data:image/png;base64,{encoded}。",
        session_key="test",
    )
    assert path and Path(path).read_bytes() == PNG
    assert "base64" in note


@pytest.mark.asyncio
async def test_structured_outputs_preserve_unescaped_paths(runtime, tmp_path):
    source = tmp_path / "result.png"
    source.write_bytes(PNG)
    runtime._find_custom_photo_tool_handler = lambda: handler
    async def handler(event, prompt):
        return {"ok": True, "outputs": [{"path": str(source)}]}
    path, note = await runtime._run_custom_tool_photo_generation("portrait", session_key="qq:FriendMessage:user", event=Event())
    assert Path(path).read_bytes() == PNG
    assert "tool_delivery_confirmed" not in note


@pytest.mark.asyncio
async def test_tool_image_send_is_archived_before_temporary_file_disappears(runtime, tmp_path):
    source = tmp_path / "temporary.png"
    source.write_bytes(PNG)
    original = Event()
    runtime._parse_message_session = lambda _: pytest.fail("real event must be reused")
    async def handler(event, prompt):
        assert event.unified_msg_origin == original.unified_msg_origin
        assert event.get_sender_id() == "user"
        event.tool_note = "same request"
        assert original.tool_note == "same request"
        await event.send(event.chain_result([Image(source.as_uri())]))
        source.unlink()
        return "ComfyUI 已生成并发送图片：" + str(source)
    runtime._find_custom_photo_tool_handler = lambda: handler
    result = await runtime._run_custom_tool_photo_generation("portrait", session_key="tool_photo_wrong:FriendMessage:other", event=original)
    assert result.generation_completed and Path(result.image_path).read_bytes() == PNG
    assert not source.exists()
    assert "tool_delivery_confirmed" not in result.note
    original.send.assert_not_awaited()
    runtime.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("umo", ["qq:FriendMessage:user", "qq:GroupMessage:group"])
async def test_synthetic_event_strips_internal_prefix_for_routing(runtime, monkeypatch, umo):
    parsed = []
    def parse(key):
        parsed.append(key)
        return SimpleNamespace(umo=key)
    monkeypatch.setattr(module, "SyntheticPrivateWakeEvent", lambda **kwargs: Event(kwargs["session"].umo))
    runtime._parse_message_session = parse
    async def handler(event, prompt):
        assert event.unified_msg_origin == umo
        return "生成失败"
    runtime._find_custom_photo_tool_handler = lambda: handler
    await runtime._run_custom_tool_photo_generation("portrait", session_key="tool_photo_" + umo)
    assert parsed == [umo]


@pytest.mark.asyncio
async def test_sent_text_alone_is_not_a_delivery_receipt(runtime, tmp_path):
    source = tmp_path / "output.png"
    source.write_bytes(PNG)
    async def handler(event, prompt):
        return "已发送图片：" + str(source)
    runtime._find_custom_photo_tool_handler = lambda: handler
    result = await runtime._run_custom_tool_photo_generation("portrait", session_key="test", event=Event())
    assert result.image_path and "tool_delivery_confirmed" not in result.note


@pytest.mark.asyncio
async def test_keyword_only_handler_executes_once(runtime):
    calls = []
    async def handler(*, event, prompt):
        calls.append(event)
        raise TypeError("failure inside generation, not argument binding")
    runtime._find_custom_photo_tool_handler = lambda: handler
    path, note = await runtime._run_custom_tool_photo_generation("portrait", session_key="test", event=Event())
    assert len(calls) == 1 and not path and "TypeError" not in note
    assert "failure inside generation" in note


@pytest.mark.asyncio
async def test_internal_typeerror_does_not_retry_positional_handler(runtime):
    calls = []
    async def handler(event, prompt):
        calls.append(prompt)
        raise TypeError("tool implementation failed")
    runtime._find_custom_photo_tool_handler = lambda: handler
    await runtime._run_custom_tool_photo_generation("portrait", session_key="test", event=Event())
    assert calls == ["portrait"]


@pytest.mark.asyncio
async def test_captured_output_survives_tool_error_after_generation(runtime, tmp_path):
    source = tmp_path / "output.png"
    source.write_bytes(PNG)
    calls = []
    async def handler(event, prompt):
        calls.append(prompt)
        await event.send(event.chain_result([Image(str(source))]))
        raise TypeError("tool cleanup failed")
    runtime._find_custom_photo_tool_handler = lambda: handler
    result = await runtime._run_custom_tool_photo_generation("portrait", session_key="test", event=Event())
    assert len(calls) == 1 and result.image_path and result.generation_completed


@pytest.mark.asyncio
async def test_failed_archive_does_not_return_temporary_path(runtime, tmp_path):
    source = tmp_path / "output.png"
    source.write_bytes(PNG)
    runtime._save_external_generated_image = AsyncMock(return_value="")
    path, note = await runtime._resolve_custom_tool_image_candidate(str(source), session_key="test")
    assert not path and "归档失败" in note


@pytest.mark.asyncio
async def test_completed_tool_failure_does_not_generate_on_another_backend(tmp_path):
    runtime = _PhotoReliabilityHarness(tmp_path)
    runtime.photo_generation_backend = "tool_call"
    runtime.custom_photo_tool_name = "comfyui_generate"
    runtime._custom_tool_photo_available = lambda: True
    async def handler(event, prompt):
        return "ComfyUI 已生成图片，但返回路径不可用：/missing/selfie.png"
    runtime._find_custom_photo_tool_handler = lambda: handler
    backend, path, note = await runtime._generate_photo_image_legacy(
        prompt_text="a quiet beach", workflow_kind="text2img", session_key="qq:FriendMessage:user", event=Event(),
    )
    assert backend == "函数工具" and not path and "归档失败" in note
    assert not runtime.external_calls
    record = runtime.data["recent_photo_generations"][0]
    assert record["generation_completed"] is True and record["failure_stage"] == "result_materialization"


@pytest.mark.asyncio
async def test_parallel_tool_events_do_not_mix_images_or_sessions(runtime, tmp_path):
    async def handler(event, prompt):
        source = tmp_path / (prompt + ".png")
        source.write_bytes(PNG + prompt.encode())
        await asyncio.sleep(0)
        await event.send(event.chain_result([Image(str(source))]))
        return "已生成并发送"
    runtime._find_custom_photo_tool_handler = lambda: handler
    results = await asyncio.gather(*[
        runtime._run_custom_tool_photo_generation(name, session_key="qq:FriendMessage:" + name, event=Event("qq:FriendMessage:" + name))
        for name in ("alice", "bob")
    ])
    assert Path(results[0].image_path).read_bytes() == PNG + b"alice"
    assert Path(results[1].image_path).read_bytes() == PNG + b"bob"
