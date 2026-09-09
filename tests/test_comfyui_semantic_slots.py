from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_ROOT))

from _runtime_loader import RUNTIME  # noqa: E402

ImageGenerationRuntime = RUNTIME.ImageGenerationRuntime
PhotoPromptSection = RUNTIME.PhotoPromptSection


def _semantic_workflow() -> dict[str, object]:
    return {
        "1": {"class_type": "Simple String", "inputs": {"text": "original flat prompt"}},
        "2": {"class_type": "AstrBot Prompt Router", "inputs": {"prompt": ["1", 0]}},
        "3": {
            "class_type": "AnimaPromptPlusClipEncode",
            "inputs": {
                "clothing_tags": ["2", 0],
                "pose_tags": ["2", 1],
                "background_tags": ["2", 2],
                "extra_prompt": ["2", 3],
            },
        },
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }


class SemanticProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ImageGenerationRuntime.__new__(ImageGenerationRuntime)
        self.sections = (
            PhotoPromptSection(
                "user_request",
                "user_request",
                positive="sitting on the bed, looking at the viewer",
                protected=True,
            ),
            PhotoPromptSection(
                "structured_wardrobe",
                "wardrobe_structure",
                positive="Concrete wardrobe specification: pale blue pajama shirt, matching pajama pants",
                protected=True,
            ),
            PhotoPromptSection(
                "scene_preset",
                "preset",
                positive="Scene preset: rainy neon street at night",
            ),
            PhotoPromptSection(
                "composition",
                "composition",
                positive="Mirror portrait composition: exactly one character in one coherent scene.",
            ),
            PhotoPromptSection(
                "subject_count",
                "composition",
                positive="Subject-count boundary: show at most one recognizable human character.",
            ),
        )

    def test_plain_selfie_projects_only_authoritative_clothing(self) -> None:
        slots = self.runtime._photo_generation_semantic_prompt_slots(
            self.sections,
            request_text="来张自拍",
        )

        self.assertEqual({"clothing_prompt"}, set(slots))
        self.assertIn("pale blue pajama shirt", slots["clothing_prompt"])
        self.assertNotIn("sitting on the bed", slots["clothing_prompt"])

    def test_explicit_pose_preset_and_supplied_slots_remain_separate(self) -> None:
        slots = self.runtime._photo_generation_semantic_prompt_slots(
            self.sections,
            request_text="来一张镜前自拍",
            requested_scene_preset="霓虹雨夜",
            supplied={
                "extra_prompt": "transparent umbrella, cinematic reflections",
                "arbitrary_node_input": "must be ignored",
                "pose_prompt": "full body, turning back toward viewer",
            },
        )

        self.assertEqual(
            {"clothing_prompt", "pose_prompt", "background_prompt", "extra_prompt"},
            set(slots),
        )
        self.assertEqual("full body, turning back toward viewer", slots["pose_prompt"])
        self.assertIn("rainy neon street", slots["background_prompt"])
        self.assertNotIn("arbitrary_node_input", slots)


class LegacyComfyUISemanticSlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_submit_applies_semantic_slots_to_memory_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow_file = root / "小爱+文本1+图片0.json"
            original = _semantic_workflow()
            workflow_file.write_text(json.dumps(original), encoding="utf-8")

            class PublicService:
                prepared_slots: dict[str, str] = {}

                def inspect_workflow(self, workflow_id):
                    self.assert_workflow_id = workflow_id
                    return {
                        "slots": [
                            {"name": "clothing_prompt"},
                            {"name": "pose_prompt"},
                            {"name": "background_prompt"},
                            {"name": "extra_prompt"},
                        ]
                    }

                def prepare_generation(self, workflow_id, slots):
                    self.prepared_slots = dict(slots)
                    modified = copy.deepcopy(original)
                    targets = {
                        "clothing_prompt": "clothing_tags",
                        "pose_prompt": "pose_tags",
                        "background_prompt": "background_tags",
                        "extra_prompt": "extra_prompt",
                    }
                    for name, value in slots.items():
                        modified["3"]["inputs"][targets[name]] = value
                    return modified, {"applied_slots": list(slots)}

            public_service = PublicService()

            class Workflow:
                submitted: dict[str, object] | None = None

                def __init__(self, server_ip, client_id):
                    self.workflow_api = {}

                def load_workflow_api(self, path):
                    self.workflow_api = json.loads(Path(path).read_text(encoding="utf-8"))

                async def submit_only(self, images, texts, videos, debug=False):
                    modified = copy.deepcopy(self.workflow_api)
                    modified["1"]["inputs"]["text"] = texts[0]
                    type(self).submitted = modified
                    return "prompt-semantic"

            async def get_result(server_ip, prompt_id):
                return "http://127.0.0.1/view/result.png", "image", []

            async def download(url):
                return str(root / "temp.png")

            async def persist(path, session_key):
                return str(root / "persistent.png")

            module = SimpleNamespace(
                _plugin_config={"debug_mode": False},
                _get_server_config=lambda config: ("127.0.0.1:8188", "client"),
                _get_workflow_dir=lambda: root,
                find_workflow_file=lambda *args: str(workflow_file),
                ComfyUIWorkflow=Workflow,
                _get_result_for_prompt=get_result,
                _download_image_to_temp=download,
                _save_image_to_persistent_path=persist,
            )
            runtime = ImageGenerationRuntime.__new__(ImageGenerationRuntime)
            runtime.comfyui_photo_wait_seconds = 1
            runtime._get_comfyui_module = lambda: module
            runtime._get_comfyui_public_service = lambda: public_service
            runtime._comfyui_renderable_prompt_text = lambda value: value
            runtime._photo_reference_prompt_for_backend_capacity = (
                lambda value, **kwargs: value
            )

            image_path, note = await runtime._run_comfyui_photo_workflow(
                "小爱",
                "unchanged flat portrait prompt",
                session_key="synthetic-private",
                semantic_prompt_slots={
                    "clothing_prompt": "red cropped jacket, black pleated skirt",
                    "pose_prompt": "one hand waving",
                    "arbitrary_node_input": "must be ignored",
                },
            )

            self.assertEqual(str(root / "persistent.png"), image_path)
            self.assertEqual("ok", note)
            self.assertEqual(
                {
                    "clothing_prompt": "red cropped jacket, black pleated skirt",
                    "pose_prompt": "one hand waving",
                },
                public_service.prepared_slots,
            )
            submitted = Workflow.submitted
            self.assertIsNotNone(submitted)
            self.assertEqual("unchanged flat portrait prompt", submitted["1"]["inputs"]["text"])
            self.assertEqual("red cropped jacket, black pleated skirt", submitted["3"]["inputs"]["clothing_tags"])
            self.assertEqual("one hand waving", submitted["3"]["inputs"]["pose_tags"])
            self.assertEqual(["2", 2], submitted["3"]["inputs"]["background_tags"])
            self.assertEqual(original, json.loads(workflow_file.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
