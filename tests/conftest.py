"""Reuse the existing AstrBot boundary doubles for standalone plugin tests."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
if importlib.util.find_spec("astrbot") is None:
    from _runtime_loader import load_runtime_module

    load_runtime_module(package_name="astrbot_plugin_image_companion", load_main=True)
