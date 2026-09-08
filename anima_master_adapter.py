# -*- coding: utf-8 -*-
"""Payload-only integration with YayiMiko/anima-master (0.9.1 contract)."""
from __future__ import annotations

import asyncio
import inspect
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

try:
    from .generation_contracts import BackendCapabilitiesV1, GenerationResultV1
except ImportError:  # pragma: no cover
    from generation_contracts import BackendCapabilitiesV1, GenerationResultV1


def find_anima_master_plugin(context: Any) -> Any | None:
    getter = getattr(context, "get_registered_star", None)
    if not callable(getter):
        return None
    for name in ("astrbot_plugin_anima_master", "anima-master", "anima_master"):
        try:
            metadata = getter(name)
            plugin = getattr(metadata, "star_cls", metadata)
            if isinstance(plugin, type) or getattr(metadata, "activated", True) is False:
                continue
            if callable(getattr(plugin, "_generate_payload", None)):
                return plugin
        except Exception:
            continue
    return None


def reference_capacity(plugin: Any) -> int:
    config = getattr(plugin, "config", {}) or {}
    enabled = config.get("img2img_enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"true", "1", "yes", "on"}
    return int(bool(enabled) and all(
        callable(getattr(plugin, name, None))
        for name in ("_run_tool", "_ensure_comfyui_ready", "_is_allowed")
    ))


class AnimaMasterAdapter:
    backend = "anima_master"

    def __init__(self, plugin: Any, event_factory: Callable[[], Any], materialize: Callable[..., Any]) -> None:
        self.plugin = plugin
        self.event_factory = event_factory
        self.materialize = materialize

    async def capabilities(self, route) -> BackendCapabilitiesV1:
        capacity = reference_capacity(self.plugin)
        return BackendCapabilitiesV1(
            text2img=True,
            edit=bool(capacity),
            negative_prompt=route.key.operation != "edit",
            max_reference_images=capacity,
            reference_roles=("identity", "outfit", "edit_source") if capacity else (),
            source="anima_master_plugin",
        )

    async def generate(self, route, spec, prompt, references, trace):
        return await self.execute(
            prompt.positive_prompt,
            negative_prompt=prompt.negative_prompt,
            operation=spec.operation,
            references=references.submitted,
            request_id=spec.request_id,
            model_profile=prompt.model_profile,
            workflow=route.key.workflow,
            trace=trace,
        )

    async def execute(
        self, positive_prompt: str, *, negative_prompt: str = "", operation: str = "text2img",
        references=(), image_size: str = "", request_id: str = "", model_profile: str = "anima",
        workflow: str = "", trace=None,
    ) -> GenerationResultV1:
        trace = list(trace or ())
        base = dict(request_id=request_id, backend=self.backend, model_profile=model_profile, workflow=workflow)

        def failure(code: str, note: str, stage: str = "submission", **extra):
            return GenerationResultV1(**base, error_code=code, note=note, failure_stage=stage, trace=tuple(trace), **extra)

        if self.plugin is None:
            return failure("route_unavailable", "Anima 绘图大师未安装、未启用或版本不兼容", "route")
        paths = [str(item.path) for item in references]
        if operation == "edit" and not paths:
            return failure("capability_missing", "Anima 改图缺少本次原图", "reference")
        if paths and not reference_capacity(self.plugin):
            return failure("capability_missing", "Anima 参考图需要在绘图大师中开启实验性 img2img_enabled", "reference")
        if any(not Path(path).is_file() for path in paths):
            return failure("capability_missing", "Anima 参考图文件不可用", "reference")
        dimensions = {}
        if image_size:
            size = re.fullmatch(r"(\d+)[xX](\d+)", image_size.strip())
            if not size:
                return failure("capability_missing", "Anima 图片尺寸请填写宽x高，例如 1024x1536", "configuration")
            dimensions = dict(width=int(size[1]), height=int(size[2]))

        submitted = ()
        degraded = []
        try:
            event = self.event_factory()
            trace.append({"stage": "anima_master_submission", "data": {"operation": operation, "reference_count": min(1, len(paths))}})
            if paths:
                if not self.plugin._is_allowed(event):
                    return failure("submission_failed", "Anima 绘图大师拒绝当前会话权限")
                ready = await self.plugin._ensure_comfyui_ready(event)
                if not isinstance(ready, dict) or not ready.get("ok"):
                    return failure("submission_failed", f"Anima ComfyUI 未就绪：{ready.get('error', 'unknown') if isinstance(ready, dict) else 'invalid_response'}")
                # Anima's resolver only accepts workspace files. Use a per-request
                # directory outside inputs so its latest-image lookup cannot reuse it.
                root = Path(self.plugin._runtime.root) / "workspace" / "companion_references"
                root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="image_", dir=root) as directory:
                    source = Path(paths[0])
                    target = Path(directory) / ("reference" + source.suffix.lower())
                    shutil.copy2(source, target)
                    payload = await self.plugin._run_tool(["edit", "--prompt", positive_prompt, "--input", str(target)])
                submitted = tuple(item.reference_id for item in references[:1])
                if negative_prompt:
                    degraded.append("negative_prompt:upstream_edit_defaults")
                if image_size:
                    degraded.append("size:upstream_reference_dimensions")
                if len(paths) > 1:
                    degraded.append(f"references:1/{len(paths)}")
            else:
                generate = self.plugin._generate_payload
                # The prepared-prompt contract prevents a second prompt optimizer
                # from replacing the companion's scene or clothing instructions.
                if "prepared_prompt" not in inspect.signature(generate).parameters:
                    return failure("route_unavailable", "请更新 Anima 绘图大师至支持 prepared_prompt 的版本（已适配 0.9.1）", "route")
                payload = await generate(
                    event, positive_prompt, prepared_prompt=positive_prompt,
                    prepared_prompt_summary={"source": "image_companion", "operation": operation},
                    negative_prompt=negative_prompt or None, **dimensions,
                )
            if not isinstance(payload, dict):
                return failure("submission_failed", "Anima 返回了无效的生成结果")
            task_id = str(payload.get("task_id") or payload.get("prompt_id") or "")
            if not payload.get("ok"):
                return failure("submission_failed", f"Anima 生成失败：{payload.get('error') or 'unknown_error'}", task_id=task_id)
            outputs = payload.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                return failure("result_materialization_failed", "Anima 已完成但未返回 outputs 图片路径", "result_materialization", generation_completed=True, task_id=task_id)
            path = str(outputs[0] or "")
            if not Path(path).is_absolute():
                path = str(Path(self.plugin._runtime.root) / path)
            try:
                materialized = await self.materialize(path)
            except Exception as exc:
                return failure(
                    "result_materialization_failed", f"Anima 已完成但图片归档失败：{type(exc).__name__}",
                    "result_materialization", generation_completed=True, task_id=task_id,
                )
            if not materialized:
                return failure("result_materialization_failed", "Anima 已完成但图片读取或归档失败", "result_materialization", generation_completed=True, task_id=task_id)
            note = "ok；已使用 1 张参考图" if submitted else "ok"
            if "negative_prompt:upstream_edit_defaults" in degraded:
                note += "；改图负面词沿用 Anima 工作流默认值"
            if "size:upstream_reference_dimensions" in degraded:
                note += "；改图尺寸由原图和 Anima max_image_side 决定"
            trace.append({"stage": "anima_master_result", "data": {"task_id": task_id, "materialized": True}})
            return GenerationResultV1(
                **base, task_id=task_id, image_path=str(materialized), note=note,
                generation_completed=True, submitted_reference_ids=submitted,
                degraded_capabilities=tuple(degraded), trace=tuple(trace),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return failure("submission_failed", f"Anima 调用失败：{type(exc).__name__}: {exc}")
