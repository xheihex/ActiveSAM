"""Run the upstream single-image methods with a PyTorch base class in Colab."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import warnings

from mmengine.registry import Registry
import torch
from tqdm.auto import tqdm

from sam3 import model_builder


_UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "activesam"
_UPSTREAM_NAMESPACE = f"{__package__}._upstream"


def _load_segmentor():
    # Use the original helper modules without running the evaluation imports
    # in activesam/__init__.py.
    package = ModuleType(_UPSTREAM_NAMESPACE)
    package.__path__ = [str(_UPSTREAM_DIR)]
    package.__package__ = _UPSTREAM_NAMESPACE
    sys.modules[_UPSTREAM_NAMESPACE] = package

    registry = Registry("activesam_colab_models")

    def import_for_segmentor(name, globals=None, locals=None, fromlist=(), level=0):
        # Only the original segmentor module sees these two bindings.
        if level == 0 and name == "mmseg.models.segmentors" and tuple(fromlist) == ("BaseSegmentor",):
            return SimpleNamespace(BaseSegmentor=torch.nn.Module)
        if level == 0 and name == "mmseg.registry" and tuple(fromlist) == ("MODELS",):
            return SimpleNamespace(MODELS=registry)
        return builtins.__import__(name, globals, locals, fromlist, level)

    name = f"{_UPSTREAM_NAMESPACE}.segmentor"
    spec = importlib.util.spec_from_file_location(name, _UPSTREAM_DIR / "segmentor.py")
    module = importlib.util.module_from_spec(spec)
    module.__builtins__ = {**vars(builtins), "__import__": import_for_segmentor}
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for module_name in tuple(sys.modules):
            if module_name == _UPSTREAM_NAMESPACE or module_name.startswith(_UPSTREAM_NAMESPACE + "."):
                del sys.modules[module_name]
        raise
    return module


_segmentor = _load_segmentor()
_prompts = sys.modules[f"{_UPSTREAM_NAMESPACE}.prompts"]
load_vocabulary = _prompts.load_vocabulary
canonicalize = _prompts.canonicalize


class _FP32Torch:
    """Disable CUDA BF16 autocast only inside the private upstream modules."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(self, device_type, dtype=None, enabled=True, **kwargs):
        if device_type == "cuda" and dtype == torch.bfloat16:
            enabled = False
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, **kwargs)


def _check_loaded_parameters(model, incompatible_keys):
    missing = incompatible_keys.missing_keys
    if missing:
        raise ValueError(f"SAM 3 权重缺少 {len(missing)} 个参数。前 10 项：{missing[:10]}")


def build_model(config):
    """Return the upstream model and whether native CUDA BF16 is enabled."""
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    torch_for_demo = torch if native_bf16 else _FP32Torch()
    _segmentor.torch = torch_for_demo
    _prompts.torch = torch_for_demo
    if not native_bf16:
        warnings.warn("当前设备缺少原生 BF16，将尝试以 FP32 运行，显存用量可能更高。", stacklevel=2)

    load_checkpoint = model_builder._load_checkpoint
    setup_device = model_builder._setup_device_and_mode
    with tqdm(
        total=4, desc="构建 SAM 3 模型",
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [已用 {elapsed}]",
    ) as progress:
        def advance(stage):
            progress.update(1)
            progress.set_description_str(stage)

        def load_checkpoint_with_progress(model, checkpoint_path):
            advance("读取并载入本地权重")
            with model.register_load_state_dict_post_hook(_check_loaded_parameters):
                return load_checkpoint(model, checkpoint_path)

        def setup_device_with_progress(model, device, eval_mode):
            advance("将模型移到 GPU")
            model = setup_device(model, device, eval_mode)
            advance("初始化类别提示与签名")
            return model

        with (
            patch.object(model_builder, "_load_checkpoint", load_checkpoint_with_progress),
            patch.object(model_builder, "_setup_device_and_mode", setup_device_with_progress),
        ):
            model = _segmentor.ActiveSAM(**config).eval()
        advance("模型加载完成")
    return model, native_bf16
