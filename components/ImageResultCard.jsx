"use client";
 
// 识别结果卡：图片 + 最可能类别 + 置信度 + 前三名。
const API = process.env.NEXT_PUBLIC_API_BASE_URL;
 
export default function ImageResultCard({ result, onOpenHistory }) {
  return (
    <article className="panel panel-half lab-panel result-panel card">
      <div className="panel-heading panel-heading-row">
        <div>
          <p className="section-kicker">结果区</p>
          <h3>识别结果</h3>
        </div>
        <button type="button" className="ghost-button" onClick={onOpenHistory}>
          历史记录
        </button>
      </div>
 
      {!result ? (
        <p className="history-empty">还没有结果，先上传一张图片试试。</p>
      ) : (
        <div className="result-stack">
          <img
            className="result-image"
            src={`${API}${result.image_url}`}
            alt="识别图片"
          />
 
          <div className="result-grid">
            <div className="result-badge">
              <span>最可能</span>
              <strong className="result-label">{result.label}</strong>
            </div>
            <div className="result-badge">
              <span>置信度</span>
              <strong>{Math.round(result.score * 100)}%</strong>
            </div>
          </div>
 
          <div className="result-item">
            <span>前 {result.top?.length ?? 0} 名</span>
            <div className="top-list">
              {result.top?.map((it) => (
                <div className="top-row" key={it.label}>
                  <span className="top-name">{it.label}</span>
                  <div className="top-bar">
                    <i style={{ width: `${it.score * 100}%` }} />
                  </div>
                  <span className="top-score">
                    {(it.score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
