"use client";
 
// 图片识别历史：缩略图 + 类别 + 置信度 + 时间。
import { useEffect } from "react";
 
const API = process.env.NEXT_PUBLIC_API_BASE_URL;
 
function formatTime(iso) {
  return new Date(iso).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
 
export default function ImageHistoryModal({ open, items, onClose }) {
  useEffect(() => {
    if (!open) return;
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);
 
  if (!open) return null;
 
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-heading">
          <div>
            <p className="section-kicker">历史记录</p>
            <h3>最近的识别</h3>
          </div>
          <button type="button" className="modal-close" onClick={onClose}>
            关闭
          </button>
        </div>
 
        {items.length === 0 ? (
          <p className="history-empty">还没有记录，先传一张图试试。</p>
        ) : (
          <div className="history-list">
            {items.map((item) => (
              <div className="history-item history-item-image" key={item.id}>
                <img src={`${API}${item.image_url}`} alt="" />
                <div>
                  <p className="history-text">
                    {item.label} · {(item.score * 100).toFixed(1)}%
                  </p>
                  <span className="history-meta">
                    {item.original_name} · {formatTime(item.created_at)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
