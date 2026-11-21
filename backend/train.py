# backend/train.py
"""微调 ResNet18 / EfficientNet-B0，做「多作物病害」分类。

用法（在 backend 目录、已装 torch 的环境里运行）：
    python train.py --data "D:\\datasets\\PlantVillage\\color" --epochs 6
    python train.py --data "D:\\datasets\\New Plant Diseases Dataset(Augmented)" --arch efficientnet_b0 --epochs 8

数据集要求（两种都支持）：
    A. ImageFolder 结构：<data>/<类别名>/xxx.jpg
    B. 已分好：<data>/train/<类别名>/...  与  <data>/valid/<类别名>/...
       有 train/ 和 valid/ 就直接用；没有就按 --val-ratio 从 A 里随机切分。

产物（写到 backend/weights/）：
    leaf.pt        state_dict，只含权重
    classes.json   类别顺序（推理时必须原样使用，别手抄改顺序）
    meta.json      网络结构、输入尺寸、归一化参数、验证准确率

训练完跑：
    python test_model.py "某张测试图.jpg"
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

# 田间照片常有不完整 JPEG，别让一张坏图中断整个训练
ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
EVAL_RESIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description="微调图像分类模型（多作物病害）")
    p.add_argument("--data", required=True,
                   help="数据集根目录（ImageFolder 结构，或含 train/ 和 valid/ 子目录）")
    p.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18",
                   help="网络结构，默认 resnet18（和手册第 3 章代码一致）")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="没有 valid/ 时，从数据里切多少比例做验证集")
    p.add_argument("--workers", type=int, default=0,
                   help="Windows 上建议 0，避免多进程读图的坑")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def safe_loader(path):
    """自己读图并转 RGB：处理灰度图、RGBA、损坏文件。"""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(EVAL_RESIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_tf, eval_tf


def build_loaders(args):
    root = Path(args.data)
    if not root.is_dir():
        raise SystemExit(f"数据集目录不存在：{root}")

    train_tf, eval_tf = build_transforms()
    train_dir, valid_dir = root / "train", root / "valid"

    if train_dir.is_dir() and valid_dir.is_dir():
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf, loader=safe_loader)
        val_ds = datasets.ImageFolder(valid_dir, transform=eval_tf, loader=safe_loader)
        if train_ds.classes != val_ds.classes:
            raise SystemExit("train/ 与 valid/ 的子目录（类别）不一致，请检查数据集")
        classes = train_ds.classes
        print(f"检测到已划分数据集：train={len(train_ds)} 张，valid={len(val_ds)} 张")
    else:
        # 同一份数据建两份（transform 不同），再用同一组随机下标切分，保证两边对应
        base_train = datasets.ImageFolder(root, transform=train_tf, loader=safe_loader)
        base_eval = datasets.ImageFolder(root, transform=eval_tf, loader=safe_loader)
        if len(base_train) == 0:
            raise SystemExit(f"{root} 下没有找到 类别子目录/图片")

        n = len(base_train)
        n_val = max(1, int(n * args.val_ratio))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=gen).tolist()

        train_ds = Subset(base_train, perm[:n_train])
        val_ds = Subset(base_eval, perm[n_train:])
        classes = base_train.classes
        print(f"随机切分数据集：train={n_train} 张，valid={n_val} 张（val-ratio={args.val_ratio}）")

    print(f"类别数：{len(classes)}")
    for i, c in enumerate(classes):
        print(f"  [{i:>2}] {c}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, drop_last=False)
    return train_loader, val_loader, classes


def build_model(arch, num_classes):
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += images.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(targets.tolist())
    return y_true, y_pred


def report(y_true, y_pred, classes):
    """逐类精确率/召回率/F1 + 最常混淆的组合（比 38x38 混淆矩阵好读）。"""
    n = len(classes)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    pairs = {}

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1
            pairs[(t, p)] = pairs.get((t, p), 0) + 1

    print("\n idx  类别                                        样本  精确率  召回率    F1")
    for i, name in enumerate(classes):
        support = tp[i] + fn[i]
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        rec = tp[i] / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        flag = ""
        if support == 0:
            flag = "  <-- 验证集里没有这个类"
        elif rec < 0.5:
            flag = "  <-- 召回率偏低"
        print(f"{i:>4}  {name:<42} {support:>5}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}{flag}")

    overall = sum(tp) / max(1, len(y_true))
    print(f"\n总体准确率：{overall * 100:.2f}%（验证样本 {len(y_true)} 张）")

    worst = sorted(pairs.items(), key=lambda kv: -kv[1])[:10]
    if worst:
        print("\n最容易混淆的 10 组（真实 → 被判成，次数）：")
        for (t, p), cnt in worst:
            print(f"  {classes[t]} → {classes[p]}  ×{cnt}")


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}    torch：{torch.__version__}    结构：{args.arch}")

    train_loader, val_loader, classes = build_loaders(args)

    model = build_model(args.arch, len(classes)).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"参数量：{num_params / 1e6:.2f} M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  * 保存为当前最优"

        print(f"[{epoch}/{args.epochs}] "
              f"train loss={train_loss:.4f} acc={train_acc * 100:.2f}% | "
              f"valid loss={val_loss:.4f} acc={val_acc * 100:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s{mark}")

    elapsed = time.time() - started
    print(f"\n训练结束，用时 {elapsed / 60:.1f} 分钟，最佳验证准确率 {best_acc * 100:.2f}%")

    if best_state is None:
        raise SystemExit("没有产生任何权重（检查数据集是否为空）")

    # 用最优权重做最终评估
    model.load_state_dict(best_state)
    y_true, y_pred = collect_predictions(model, val_loader, device)
    report(y_true, y_pred, classes)

    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt_path = WEIGHTS_DIR / "leaf.pt"
    classes_path = WEIGHTS_DIR / "classes.json"
    meta_path = WEIGHTS_DIR / "meta.json"

    torch.save(best_state, ckpt_path)
    classes_path.write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "arch": args.arch,
        "classes": classes,
        "num_classes": len(classes),
        "input_size": INPUT_SIZE,
        "eval_resize": EVAL_RESIZE,
        "mean": MEAN,
        "std": STD,
        "val_acc": round(best_acc, 4),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n已保存：")
    print(f"  {ckpt_path}")
    print(f"  {classes_path}")
    print(f"  {meta_path}")
    print('\n下一步：python test_model.py "某张测试图.jpg"')
    print("model_service.py 会自动检测到 leaf.pt 并切换成你的模型，接口不用改。")


if __name__ == "__main__":
    main()
