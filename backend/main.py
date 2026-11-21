# backend/main.py
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
 
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pypinyin import Style, lazy_pinyin
from snownlp import SnowNLP
 
import model_service
from storage import (
    get_history,
    get_predictions,
    init_db,
    save_prediction,
    save_record,
)
 
load_dotenv()
 
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
 
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
 
MAX_UPLOAD_BYTES = 5 * 1024 * 1024           # 5 MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
 
init_db()
 
app = FastAPI(title="Zero to Tech API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_credentials=True,
)
 
# 把 uploads/ 挂成静态目录：浏览器访问 /uploads/xxx.jpg 就能拿到图片
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
 
 
def get_session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get("session_id")
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie(
            "session_id", sid,
            httponly=True, samesite="lax",
            max_age=60 * 60 * 24 * 30,
        )
    return sid
 
 
# ---------------- 原有：个人主页 ----------------
profile = home = {
    "heroTitle": "关于我(来自后端)",
    "heroSubtitle": "项目，创意，灵感，心得，我的作品",
    "featuredWork": {
        "kicker": "作品",
        "title": "文字实验室",
        "copy": "拼音和情绪，挖掘中文里的细节",
        "linkLabel": "打开作品",
    },
    "identity": {
        "motto": "已识乾坤大，尤怜草木青",
        "learning": "零到全栈",
    },
}
 
 
@app.get("/api/profile")
def get_profile():
    return profile
 
 
# ---------------- 原有：文字分析 ----------------
class AnalyzeRequest(BaseModel):
    text: str
 
 
def score_label(score):
    if score >= 0.6:
        return "偏积极"
    elif score <= 0.4:
        return "偏消极"
    else:
        return "中性"
 
 
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request, response: Response):
    sid = get_session_id(request, response)
    text = req.text
    score = round(SnowNLP(text).sentiments, 2)
    result = {
        "text": text,
        "score": score,
        "label": score_label(score),
        "pinyin": " ".join(lazy_pinyin(text, style=Style.TONE)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_record(sid, result)
    return result
 
 
@app.get("/api/history")
def history(request: Request, response: Response, limit: int = 10):
    sid = get_session_id(request, response)
    return get_history(sid, limit)
 
 
# ---------------- 新增：图片识别 ----------------
@app.post("/api/predict")
async def predict_image(
    request: Request,
    response: Response,
    file: UploadFile = File(...),          # 字段名必须叫 file
):
    sid = get_session_id(request, response)
 
    # 1. 扩展名白名单
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "只支持 JPG / PNG / WebP / BMP 图片")
 
    # 2. 读出内容，检查大小
    data = await file.read()
    if not data:
        raise HTTPException(400, "上传的文件是空的")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片太大，请压缩到 5 MB 以内")
 
    # 3. 不光看后缀，用 Pillow 验证它真的是图片
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "这个文件不是有效的图片")
 
    # 4. 用 uuid 重命名后存盘（绝不使用用户给的文件名做路径）
    stored_name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / stored_name).write_bytes(data)
 
    # 5. 调用模型
    try:
        top = model_service.predict(data, top_k=3)
    except Exception as e:
        (UPLOAD_DIR / stored_name).unlink(missing_ok=True)   # 失败就删图，避免垃圾文件
        raise HTTPException(500, f"模型推理失败：{e}")
 
    best = top[0]
    record = {
        "original_name": file.filename or stored_name,
        "stored_name": stored_name,
        "image_url": f"/uploads/{stored_name}",
        "label": best["label"],
        "score": best["score"],
        "top": top,
        "model_name": model_service.MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
 
    # 6. 写库
    save_prediction(sid, record)
    return record
 
 
@app.get("/api/predictions")
def predictions(request: Request, response: Response, limit: int = 10):
    sid = get_session_id(request, response)
    rows = get_predictions(sid, limit)
    for r in rows:
        r["image_url"] = f"/uploads/{r['stored_name']}"
    return rows
 
 
@app.get("/api/health")
def health():
    return {"ok": True, "model": model_service.MODEL_NAME}

