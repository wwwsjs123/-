# backend/convert_hf_mobilenet.py
"""把 HuggingFace(transformers) 格式的 MobileNetV2 权重，转成 torchvision 格式的 leaf.pt。

为什么需要这一步：同一个网络，两个库给同一层的「名字」不一样
    HF 版：        mobilenet_v2.conv_stem.first_conv.convolution.weight
    torchvision：  features.0.0.weight
权重本身是通用的，只是编号表不同。转换完成后就是普通 pt 权重，用 torchvision 加载，
不依赖 transformers / tensorflow。

用法：
    python convert_hf_mobilenet.py

产物（都写到 backend/weights/）：
    leaf.pt         torchvision 格式的 state_dict（38 类）
    classes.json    类别名（英文，顺序与 config.json 的 id2label 完全一致）
    classes_zh.json 类别名（中文，给页面显示用，顺序同上）
    meta.json       结构 / 输入尺寸 / 归一化 / 来源信息
"""

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import mobilenet_v2

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"

SRC_BIN = WEIGHTS / "hf_mobilenet_plantvillage.bin"
SRC_CFG = WEIGHTS / "hf_config.json"

MODEL_URL = (
    "https://huggingface.co/linkanjarad/"
    "mobilenet_v2_1.0_224-plant-disease-identification"
)

# 预处理参数来自该模型的 preprocessor_config.json：
# resize shortest_edge=256 -> center crop 224，像素缩放到 [0,1] 后用 mean=0.5/std=0.5 归一化（即 [-1,1]）
INPUT_SIZE = 224
EVAL_RESIZE = 256
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# 中文名（顺序必须和下面 classes 一致；想改措辞可以改这里，但不要动顺序）
ZH = {
    "Apple Scab": "苹果黑星病",
    "Apple with Black Rot": "苹果黑腐病",
    "Cedar Apple Rust": "苹果锈病（雪松-苹果锈病）",
    "Healthy Apple": "健康苹果",
    "Healthy Blueberry Plant": "健康蓝莓",
    "Cherry with Powdery Mildew": "樱桃白粉病",
    "Healthy Cherry Plant": "健康樱桃",
    "Corn (Maize) with Cercospora and Gray Leaf Spot": "玉米灰斑病（尾孢菌叶斑）",
    "Corn (Maize) with Common Rust": "玉米普通锈病",
    "Corn (Maize) with Northern Leaf Blight": "玉米北方叶枯病",
    "Healthy Corn (Maize) Plant": "健康玉米",
    "Grape with Black Rot": "葡萄黑腐病",
    "Grape with Esca (Black Measles)": "葡萄埃斯卡病（黑麻疹病）",
    "Grape with Isariopsis Leaf Spot": "葡萄褐斑病（Isariopsis 叶斑）",
    "Healthy Grape Plant": "健康葡萄",
    "Orange with Citrus Greening": "柑橘黄龙病",
    "Peach with Bacterial Spot": "桃细菌性斑点病",
    "Healthy Peach Plant": "健康桃",
    "Bell Pepper with Bacterial Spot": "甜椒细菌性斑点病",
    "Healthy Bell Pepper Plant": "健康甜椒",
    "Potato with Early Blight": "马铃薯早疫病",
    "Potato with Late Blight": "马铃薯晚疫病",
    "Healthy Potato Plant": "健康马铃薯",
    "Healthy Raspberry Plant": "健康树莓",
    "Healthy Soybean Plant": "健康大豆",
    "Squash with Powdery Mildew": "南瓜白粉病",
    "Strawberry with Leaf Scorch": "草莓叶焦病",
    "Healthy Strawberry Plant": "健康草莓",
    "Tomato with Bacterial Spot": "番茄细菌性斑点病",
    "Tomato with Early Blight": "番茄早疫病",
    "Tomato with Late Blight": "番茄晚疫病",
    "Tomato with Leaf Mold": "番茄叶霉病",
    "Tomato with Septoria Leaf Spot": "番茄斑枯病（Septoria 叶斑）",
    "Tomato with Spider Mites or Two-spotted Spider Mite": "番茄红蜘蛛危害",
    "Tomato with Target Spot": "番茄靶斑病",
    "Tomato Yellow Leaf Curl Virus": "番茄黄化曲叶病毒病",
    "Tomato Mosaic Virus": "番茄花叶病毒病",
    "Healthy Tomato Plant": "健康番茄",
}


def translate(key, expand_indices):
    """HF 的键名 -> torchvision 的键名；映射不了返回 None。"""
    # 1) 主干第一段：first_conv / conv_3x3 / reduce_1x1 其实横跨 torchvision 的 features.0 和 features.1
    m = re.match(
        r"^mobilenet_v2\.conv_stem\.(first_conv|conv_3x3|reduce_1x1)\."
        r"(convolution|normalization)\.(.+)$",
        key,
    )
    if m:
        part, kind, param = m.groups()
        conv_name, bn_name = {
            "first_conv": ("features.0.0", "features.0.1"),
            "conv_3x3": ("features.1.conv.0.0", "features.1.conv.0.1"),
            "reduce_1x1": ("features.1.conv.1", "features.1.conv.2"),
        }[part]
        return f"{conv_name if kind == 'convolution' else bn_name}.{param}"

    # 2) 主体倒残差块：HF layer.i -> torchvision features.(i+2)
    m = re.match(
        r"^mobilenet_v2\.layer\.(\d+)\.(expand_1x1|conv_3x3|reduce_1x1)\."
        r"(convolution|normalization)\.(.+)$",
        key,
    )
    if m:
        idx = int(m.group(1))
        part, kind, param = m.group(2), m.group(3), m.group(4)
        block = idx + 2

        # 有 expand 的块，torchvision 的 conv 顺序是 [expand, depthwise, linear, bn]
        # 没有 expand 的块（expand_ratio=1）是 [depthwise, linear, bn]
        if idx in expand_indices:
            slots = {
                "expand_1x1": ("conv.0.0", "conv.0.1"),
                "conv_3x3": ("conv.1.0", "conv.1.1"),
                "reduce_1x1": ("conv.2", "conv.3"),
            }
        else:
            slots = {
                "conv_3x3": ("conv.0.0", "conv.0.1"),
                "reduce_1x1": ("conv.1", "conv.2"),
            }
        if part not in slots:
            return None
        conv_name, bn_name = slots[part]
        return f"features.{block}.{conv_name if kind == 'convolution' else bn_name}.{param}"

    # 3) 最后的 1x1 卷积（320 -> 1280）
    m = re.match(r"^mobilenet_v2\.conv_1x1\.(convolution|normalization)\.(.+)$", key)
    if m:
        kind, param = m.groups()
        return f"{'features.18.0' if kind == 'convolution' else 'features.18.1'}.{param}"

    # 4) 分类头（HF: classifier / torchvision: classifier.1）
    m = re.match(r"^classifier\.(.+)$", key)
    if m:
        return f"classifier.1.{m.group(1)}"

    return None


def sanity_check(model, classes):
    """用 backend/samples 里的一张真实病叶图跑一次，确认转换后的数值是对的。"""
    samples = sorted((HERE / "samples").glob("*.*"))
    if not samples:
        print("[提示] backend/samples/ 里没有图片，跳过数值校验")
        return

    path = samples[0]
    tf = transforms.Compose([
        transforms.Resize(EVAL_RESIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")

    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0))[0], dim=0)

    values, indices = probs.topk(3)
    print(f"\n[数值校验] {path.name}")
    for v, i in zip(values, indices):
        print(f"  {classes[int(i)]:<45} {float(v) * 100:5.1f}%")
    print("  ↑ 如果这张（苹果黑星病）图排在第一位，说明键名映射正确")


def main():
    if not SRC_BIN.is_file():
        raise SystemExit(f"找不到 HF 权重：{SRC_BIN}")
    if not SRC_CFG.is_file():
        raise SystemExit(f"找不到配置文件：{SRC_CFG}")

    hf = torch.load(SRC_BIN, map_location="cpu", weights_only=True)
    cfg = json.loads(SRC_CFG.read_text(encoding="utf-8"))

    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    num_classes = len(id2label)
    classes = [id2label[i] for i in range(num_classes)]
    print(f"源权重：{len(hf)} 个张量，{num_classes} 类，模型结构 {cfg.get('architectures')}")

    expand_indices = {
        int(m.group(1))
        for k in hf
        if (m := re.match(r"^mobilenet_v2\.layer\.(\d+)\.expand_1x1\.", k))
    }
    print(f"含 expand 的倒残差块：{len(expand_indices)} 个（其余块 expand_ratio=1）")

    model = mobilenet_v2(num_classes=num_classes)
    target = model.state_dict()

    new_sd = {}
    unmapped = []
    for key, value in hf.items():
        new_key = translate(key, expand_indices)
        if new_key is None:
            unmapped.append(key)
            continue
        new_sd[new_key] = value

    if unmapped:
        raise SystemExit("有键无法映射（需要补规则）：\n  " + "\n  ".join(unmapped))

    # 三重校验：键全、形状一致、strict 加载成功
    missing = sorted(set(target) - set(new_sd))
    unexpected = sorted(set(new_sd) - set(target))
    if missing or unexpected:
        raise SystemExit(f"键不匹配\n缺失({len(missing)})：{missing[:10]}\n多余({len(unexpected)})：{unexpected[:10]}")

    for key, value in new_sd.items():
        if tuple(target[key].shape) != tuple(value.shape):
            raise SystemExit(f"形状不一致：{key} 期望 {tuple(target[key].shape)} 实际 {tuple(value.shape)}")

    model.load_state_dict(new_sd, strict=True)
    model.eval()
    print(f"strict=True 加载成功，参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    sanity_check(model, classes)

    # 保存
    WEIGHTS.mkdir(exist_ok=True)
    torch.save(new_sd, WEIGHTS / "leaf.pt")
    (WEIGHTS / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (WEIGHTS / "classes_zh.json").write_text(
        json.dumps([ZH.get(c, c) for c in classes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "arch": "mobilenet_v2",
        "input_size": INPUT_SIZE,
        "eval_resize": EVAL_RESIZE,
        "mean": MEAN,
        "std": STD,
        "classes": classes,
        "classes_zh": [ZH.get(c, c) for c in classes],
        "num_classes": num_classes,
        "source": MODEL_URL,
        "converted_by": "convert_hf_mobilenet.py",
    }
    (WEIGHTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n已保存：")
    print(f"  {WEIGHTS / 'leaf.pt'}")
    print(f"  {WEIGHTS / 'classes.json'} / classes_zh.json")
    print(f"  {WEIGHTS / 'meta.json'}")
    print('\n下一步：python test_model.py "某张叶子照片.jpg"')


if __name__ == "__main__":
    main()
