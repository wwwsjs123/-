"use client";

// 图片上传卡：选择 / 拖拽 / 预览 / 上传。
// 关键点：用 <label> 包住 <input type="file">，点击区域由浏览器原生触发文件框，
// 不依赖 JS 的 input.click()。这样即使页面脚本/hydration 出问题，选图依然可用。
import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE_URL;
const MAX_MB = 5;
const OK_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function ImageInputCard({ onResult }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // file 变了就生成新的预览地址；旧地址在 cleanup 里释放，避免内存泄漏
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function pick(f) {
    setError("");
    if (!f) return;

    if (!OK_TYPES.includes(f.type)) {
      setError("只支持 JPG / PNG / WebP / BMP 图片");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`图片不能超过 ${MAX_MB} MB`);
      return;
    }
    setFile(f);
  }

  async function handleUpload() {
    if (!file) {
      setError("请先选一张图片");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);   // 字段名必须和后端 File(...) 一致

      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        credentials: "include",
        body: form,                // 不要手动设 Content-Type
      });

      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(body.detail || `识别失败：${res.status}`);
      }

      onResult(body);
    } catch (e) {
      setError(e.message || "网络出错了，请确认后端已启动");
    } finally {
      setLoading(false);
    }
  }

  return (
    <article className="panel panel-half lab-panel card">
      <div className="panel-heading">
        <p className="section-kicker">输入区</p>
        <h3>上传一张叶片照片</h3>
      </div>

      <label
        className={"upload-zone" + (dragging ? " is-dragging" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        {preview ? (
          <img className="upload-preview" src={preview} alt="待识别的图片" />
        ) : (
          <span className="upload-hint">
            点击选择，或把图片拖到这里
            <br />
            <span>JPG / PNG / WebP / BMP，不超过 {MAX_MB} MB</span>
          </span>
        )}

        <input
          ref={inputRef}
          className="upload-input"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={(e) => {
            pick(e.target.files?.[0]);
            e.target.value = "";     // 允许连续选同一张图
          }}
        />
      </label>

      {file && (
        <p className="lab-count">
          {file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
        </p>
      )}

      {error && <p className="lab-error">{error}</p>}

      <button
        className="primary-button"
        type="button"
        onClick={handleUpload}
        disabled={loading}
      >
        {loading ? "识别中…" : "开始识别"}
      </button>
    </article>
  );
}
