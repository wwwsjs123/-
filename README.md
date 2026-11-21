# 作物病害识别（Crop Disease Recognition）

上传一张叶片照片，自动识别是哪种作物、得了什么病。前后端分离：Next.js 静态导出 + FastAPI + SQLite + PyTorch。

- 前端：Next.js 15 / React 19（`output: "export"` 静态导出）
- 后端：FastAPI + SQLite
- 模型：MobileNetV2（38 类，覆盖 14 种作物的健康/病害状态），权重已随仓库提供

## 功能

| 功能 | 说明 |
|---|---|
| 图片上传 | 点选或拖拽 JPG / PNG / WebP / BMP，最大 5 MB，选完立即本地预览 |
| 病害识别 | 后端调用模型推理，返回最可能类别 + 置信度 + Top3 |
| 记录保存 | 图片存 `backend/uploads/`（uuid 命名），元数据写 `backend/history.db` |
| 历史记录 | 弹窗展示缩略图、类别、置信度、时间，按浏览器 cookie 里的 session 隔离 |

## 目录结构

```
binghai/
├── app/                     # Next.js App Router
│   ├── layout.jsx           # 全站外壳 + CSS 引入
│   ├── page.jsx             # 首页
│   ├── text-lab/page.jsx    # 文字实验室（保留原功能）
│   └── image-lab/page.jsx   # 病害识别页
├── components/              # 页面组件（Image* 为识别相关）
├── css/                     # 样式，image-lab.css 为识别页专用
├── data/site.js             # 文案
├── backend/
│   ├── main.py              # FastAPI 入口：/api/profile、/api/analyze、/api/history、/api/predict、/api/predictions
│   ├── model_service.py     # 「标准插座」：MODEL_NAME + predict(bytes) -> [{label, score}]
│   ├── storage.py           # SQLite：history 表 + predictions 表
│   ├── train.py             # 自己微调 ResNet18 / EfficientNet-B0 / MobileNetV2
│   ├── test_model.py        # 命令行单测模型
│   ├── convert_hf_mobilenet.py  # 把 HF 格式权重转成 torchvision 格式
│   └── weights/
│       ├── leaf.pt          # 38 类病害分类权重（torchvision 格式 state_dict）
│       ├── classes.json     # 英文类别名（顺序即模型输出索引）
│       ├── classes_zh.json  # 中文类别名（页面显示用）
│       ├── meta.json        # 结构 / 输入尺寸 / 归一化参数
│       └── hf_config.json   # 原始 HF 模型的类别表（转换用）
├── next.config.mjs
└── package.json
```

## 快速开始

### 1. 后端

```powershell
cd binghai\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow

copy .env.example .env      # 按需修改 ALLOWED_ORIGINS
fastapi dev main.py         # http://localhost:8000/docs
```

用 GPU 的话把最后一条 `pip install` 换成对应的 CUDA 版本即可（`model_service.py` 会自动检测 `cuda`）。

### 2. 前端

```powershell
cd binghai
npm install
copy .env.example .env.local
npm run dev                 # http://localhost:3000/image-lab
```

> 前端和后端必须用同一个主机名访问：都用 `localhost` 或都用 `127.0.0.1`，否则会被浏览器的 CORS 拦住。

### 3. 只测模型（不启服务）

```powershell
cd binghai\backend
python test_model.py "某张叶片照片.jpg"
python test_model.py "图1.jpg" "图2.jpg"
```

输出形如：

```
模型：leaf-mobilenet_v2-v1（自训练）    设备：cuda
[model_service] 已加载自训练模型：...\weights\leaf.pt（mobilenet_v2，38 类）
{'label': '苹果黑星病', 'score': 0.9907}
...
```

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查，返回当前模型名 |
| POST | `/api/predict` | `multipart/form-data`，字段名必须是 `file`；返回 `label / score / top / image_url / model_name / created_at` |
| GET | `/api/predictions?limit=10` | 当前会话的识别历史 |
| GET | `/uploads/<文件名>` | 直接访问上传的原图 |
| POST | `/api/analyze` | 文字情感分析（保留的原有功能） |
| GET | `/api/history?limit=10` | 文字分析历史 |

上传校验三层：扩展名白名单 → 5 MB 限制 → `Pillow.verify()` 校验真实内容。文件名统一用 uuid 重命名，原文件名只用于展示。

## 模型说明

- 结构：MobileNetV2，输入 `224×224`，预处理 `Resize(256) → CenterCrop(224)` + `Normalize(mean=0.5, std=0.5)`（即 `[-1, 1]`），**不是** ImageNet 的均值方差。
- 类别：38 类，覆盖苹果、蓝莓、樱桃、玉米、葡萄、柑橘、桃、甜椒、马铃薯、树莓、大豆、南瓜、草莓、番茄的健康与常见病害。
- 来源：由 Hugging Face 上的 `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`（PyTorch/transformers 格式）经 `convert_hf_mobilenet.py` 转换而来。
- 局限：训练数据是 PlantVillage（实验室白底摆拍），对**田间实拍、远景、逆光、整株**照片准确率会明显下降；这类模型只能回答「像哪种病」，不能定位病斑。

### 换成自己的模型

三种方式，`main.py` 和前端都不用改：

1. 用 `train.py` 微调，产物直接覆盖 `weights/leaf.pt + classes.json + meta.json`：
   ```powershell
   python train.py --data "D:\datasets\New Plant Diseases Dataset(Augmented)" --epochs 6
   ```
2. 拿现成的 `.pt` 权重：把文件放进 `backend/weights/`，改 `model_service.py` 里的 `CUSTOM_CKPT`，并保证 `_build_arch()` 的结构与权重一致、`meta.json` 的类别顺序与训练时一致。
3. 换别的 HF 模型：替换 `weights/hf_mobilenet_plantvillage.bin` 与 `hf_config.json`，按需调整 `convert_hf_mobilenet.py` 的键名映射后重跑。

没有 `leaf.pt` 时，`model_service.py` 会自动退回 ImageNet 的 ResNet18 演示模型（能认猫狗，不认病害），方便先验证链路。

## 环境变量

| 文件 | 变量 | 说明 |
|---|---|---|
| `.env.local`（前端） | `NEXT_PUBLIC_API_BASE_URL` | 后端地址，如 `http://localhost:8000`；**构建时**写死，改完要重新 `npm run build` |
| `backend/.env` | `ALLOWED_ORIGINS` | CORS 白名单，逗号分隔，如 `http://localhost:3000,http://127.0.0.1:3000` |

## 常见问题

- **点了上传区没反应 / 控制台报 `Hydration failed`**：先关掉浏览器翻译类扩展，或用无痕窗口打开（无痕默认禁用扩展）。代码侧已把上传区改成 `<label>` 原生触发，并且 `app/layout.jsx` 的 `<html>` 加了 `suppressHydrationWarning`。
- **`422 field required: file`**：`FormData` 的字段名不是 `file`，或后端缺 `python-multipart`。
- **`/uploads/xxx 404`**：`app.mount("/uploads", StaticFiles(...))` 没生效，或 `UPLOAD_DIR` 指向不对。
- **历史是空的**：`fetch` 少了 `credentials: "include"`，或后端 `allow_credentials` 没开。
- **第一次识别很慢**：模型首次加载需要 1～10 秒，属正常；可在 `main.py` 用 `lifespan` 调 `model_service.warmup()` 预热。

## 已知事项

- 图片存本地磁盘、数据库只存路径与元数据，部署到多实例时要改成对象存储。
- `history.db` 和 `uploads/` 不进版本库（见 `.gitignore`），部署时需要用卷挂载持久化。
