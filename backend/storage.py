# backend/storage.py
import json
import sqlite3
from pathlib import Path
 
# 用绝对路径：不管你从哪个目录启动，都指向 backend/history.db
DB_FILE = Path(__file__).resolve().parent / "history.db"
 
 
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row       # 查询结果可以按字段名取
    return conn
 
 
def init_db():
    conn = get_conn()
    cur = conn.cursor()
 
    # ---------- 原有的文字分析历史 ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        text TEXT,
        score REAL,
        label TEXT,
        pinyin TEXT,
        created_at TEXT
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_session_created "
        "ON history(session_id, created_at)"
    )
 
    # ---------- 新增：图片识别记录 ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        original_name TEXT,
        stored_name TEXT,
        label TEXT,
        score REAL,
        top TEXT,
        model_name TEXT,
        created_at TEXT
    )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_session_created "
        "ON predictions(session_id, created_at)"
    )
 
    conn.commit()
    conn.close()
 
 
# ---------- 原有：文字分析 ----------
def save_record(session_id, record):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (session_id, text, score, label, pinyin, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [session_id, record["text"], record["score"],
         record["label"], record["pinyin"], record["created_at"]],
    )
    conn.commit()
    conn.close()
 
 
def get_history(session_id, limit):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM history WHERE session_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        [session_id, limit],
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
 
 
# ---------- 新增：图片识别 ----------
def save_prediction(session_id, record):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions "
        "(session_id, original_name, stored_name, label, score, top, model_name, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [session_id, record["original_name"], record["stored_name"],
         record["label"], record["score"],
         json.dumps(record["top"], ensure_ascii=False),
         record["model_name"], record["created_at"]],
    )
    conn.commit()
    conn.close()
 
 
def get_predictions(session_id, limit):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE session_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        [session_id, limit],
    ).fetchall()
    conn.close()
 
    records = []
    for row in rows:
        item = dict(row)
        item["top"] = json.loads(item["top"] or "[]")
        records.append(item)
    return records


if __name__ == "__main__":
    # 直接运行本文件时的自检：建表 + 打印两张表的结构
    # 用法：python storage.py
    init_db()
    print(f"数据库文件：{DB_FILE}")
    print(f"文件存在：{Path(DB_FILE).exists()}")

    conn = get_conn()
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    print(f"表：{tables}")

    for table in ("history", "predictions"):
        if table not in tables:
            print(f"  [缺失] {table}")
            continue
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        print(f"\n{table}（{count} 行）字段：")
        for col in cols:
            print(f"  {col['name']:<16} {col['type']:<10} notnull={col['notnull']} pk={col['pk']}")

    conn.close()
