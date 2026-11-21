# backend/test_model.py
"""命令行测试脚本：在接接口之前单独验证模型能不能用。

用法：
    python test_model.py 图片路径
    python test_model.py 图片1 图片2 图片3      # 支持一次传多张

必须在 backend 目录下运行（否则 import model_service 会失败）。
"""

import sys
from pathlib import Path

import model_service


def main(argv):
    # Windows 控制台遇到无法编码的字符时不至于直接崩掉
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    if not argv:
        print("用法：python test_model.py 图片路径 [更多图片...]")
        return 1

    print(f"模型：{model_service.MODEL_NAME}    设备：{model_service.DEVICE}")

    exit_code = 0
    for arg in argv:
        path = Path(arg)
        print(f"\n=== {path} ===")

        if not path.is_file():
            print(f"[跳过] 文件不存在：{path}")
            exit_code = 1
            continue

        try:
            results = model_service.predict(path.read_bytes())
        except Exception as exc:
            print(f"[失败] {type(exc).__name__}: {exc}")
            exit_code = 1
            continue

        if not results:
            print("[空结果] 模型没有返回任何类别")
            continue

        for item in results:
            print(item)
        best = results[0]
        print(f"最可能：{best['label']}（{best['score'] * 100:.1f}%）")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
