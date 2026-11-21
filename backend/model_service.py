# backend/model_service.py
"""模型服务：把「图片字节」变成「识别结果」。

自动切换两种工作模式（main.py 完全不用改）：
  1) 自有模型模式：backend/weights/ 里有 train.py 的产物
        leaf.pt + meta.json（或 classes.json）
     → 用你自己训练的模型，类别名从 meta/classes 里读
  2) 演示模式：上面那些文件不存在
     → 用 resnet18 的 ImageNet 权重（优先读 backend/weights/resnet18-f37072fd.pth，离线可用）

换模型只改这个文件；main.py 只通过 MODEL_NAME 和 predict() 与它打交道。
"""

import io
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
)

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"

# train.py 的产物
CUSTOM_CKPT = WEIGHTS_DIR / "leaf.pt"
CUSTOM_META = WEIGHTS_DIR / "meta.json"
CUSTOM_CLASSES = WEIGHTS_DIR / "classes.json"

# 演示模型权重（可用环境变量 MODEL_WEIGHTS 指向别处）
DEMO_CKPT = Path(
    os.getenv("MODEL_WEIGHTS") or (WEIGHTS_DIR / "resnet18-f37072fd.pth")
)

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# 有 GPU 就用 GPU，没有就用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_custom_meta():
    """读 train.py 写的 meta.json；没有就拿 classes.json 兜底。"""
    if not CUSTOM_CKPT.is_file():
        return None

    if CUSTOM_META.is_file():
        meta = json.loads(CUSTOM_META.read_text(encoding="utf-8"))
    elif CUSTOM_CLASSES.is_file():
        meta = {"classes": json.loads(CUSTOM_CLASSES.read_text(encoding="utf-8"))}
    else:
        print("[model_service] 发现 leaf.pt 但缺少 meta.json / classes.json，继续用演示模型")
        return None

    if not meta.get("classes"):
        print("[model_service] meta.json 里没有 classes，继续用演示模型")
        return None

    meta.setdefault("arch", os.getenv("MODEL_ARCH", "resnet18"))
    meta.setdefault("input_size", 224)
    meta.setdefault("eval_resize", 256)
    meta.setdefault("mean", DEFAULT_MEAN)
    meta.setdefault("std", DEFAULT_STD)
    return meta


_CUSTOM = _read_custom_meta()

if _CUSTOM:
    # 有中文类别名就优先用中文（只影响显示，索引顺序与训练一致）
    CLASSES = list(_CUSTOM.get("classes_zh") or _CUSTOM["classes"])
    MODEL_NAME = f"leaf-{_CUSTOM['arch']}-v1（自训练）"
else:
    CLASSES = ResNet18_Weights.DEFAULT.meta["categories"]
    MODEL_NAME = "resnet18-imagenet（演示模型）"

_model = None
_tf = None


def _build_arch(arch: str, num_classes: int) -> nn.Module:
    """按结构名搭出网络骨架（随机权重），分类头改成自己的类别数。"""
    if arch == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(
        f"不支持的 arch：{arch}（可选 resnet18 / efficientnet_b0 / mobilenet_v2）"
    )


def _build_custom() -> nn.Module:
    model = _build_arch(str(_CUSTOM["arch"]), len(CLASSES))
    state = torch.load(CUSTOM_CKPT, map_location="cpu", weights_only=True)
    # 兼容训练脚本多包一层 {"state_dict": ...} 的情况
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print(
        f"[model_service] 已加载自训练模型：{CUSTOM_CKPT}"
        f"（{_CUSTOM['arch']}，{len(CLASSES)} 类）"
    )
    return model.to(DEVICE).eval()


def _build_demo() -> nn.Module:
    if DEMO_CKPT.is_file() and DEMO_CKPT.stat().st_size > 0:
        try:
            state = torch.load(DEMO_CKPT, map_location="cpu", weights_only=True)
            model = resnet18(weights=None)
            model.load_state_dict(state)
            print(f"[model_service] 已加载本地演示权重：{DEMO_CKPT}")
            return model.to(DEVICE).eval()
        except Exception as exc:  # 文件损坏 / 不是 resnet18
            print(f"[model_service] 本地演示权重不可用（{type(exc).__name__}: {exc}），改用在线权重")

    print("[model_service] 尝试下载 torchvision 预训练权重（首次约 45 MB，需要联网）…")
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(DEVICE).eval()


def _load():
    """第一次调用时加载模型和预处理，之后复用（别每次请求都重新加载）。"""
    global _model, _tf
    if _model is None:
        if _CUSTOM:
            _model = _build_custom()
            input_size = int(_CUSTOM["input_size"])
            eval_resize = int(_CUSTOM["eval_resize"])
            _tf = transforms.Compose([
                transforms.Resize(eval_resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CUSTOM["mean"], std=_CUSTOM["std"]),
            ])
        else:
            _model = _build_demo()
            _tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ])
    return _model, _tf


def warmup():
    """启动时预热：让第一个真实请求不用等模型加载。"""
    _load()


@torch.no_grad()
def predict(image_bytes: bytes, top_k: int = 3):
    """图片字节 -> [{label, score}, ...]，第一条最可能。"""
    if not image_bytes:
        raise ValueError("图片内容为空")

    model, tf = _load()

    # 用 with 打开保证句柄释放；
    # exif_transpose 摆正手机竖拍照片；convert("RGB") 统一三通道并真正读入内存。
    with Image.open(io.BytesIO(image_bytes)) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")

    x = tf(img).unsqueeze(0).to(DEVICE)   # [3,224,224] -> [1,3,224,224]
    logits = model(x)[0]
    probs = torch.softmax(logits, dim=0)

    values, indices = probs.topk(min(top_k, len(CLASSES)))
    return [
        {"label": CLASSES[int(i)], "score": round(float(v), 4)}
        for v, i in zip(values, indices)
    ]
