from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import json
import os
import asyncio

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"

class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith('.html'):
            response.headers['Content-Type'] = 'text/html; charset=utf-8'
        elif path.endswith('.css'):
            response.headers['Content-Type'] = 'text/css; charset=utf-8'
        elif path.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
        return response

app.mount("/static", CustomStaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    index_path = static_dir / "index.html"
    return FileResponse(index_path, media_type="text/html; charset=utf-8")

bot_instance = None

def init_bot():
    from miniclaw.bot import miniclaw
    global bot_instance
    if bot_instance is None:
        bot_instance = miniclaw.from_config()

@app.on_event("startup")
async def startup_event():
    init_bot()

@app.post("/webhook/msg")
async def webhook_msg(request: Request):
    body = await request.json()
    message = body.get("message", "")
    session_key = body.get("session_key", "default")

    if not bot_instance:
        init_bot()

    try:
        result = bot_instance.run(message, session_key=session_key)
        if asyncio.iscoroutine(result):
            result = await result
        return {"result": str(result)}

    except Exception as e:
        return {"error": str(e)}

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"success": True, "filename": file.filename}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/sessions")
async def get_sessions():
    if not bot_instance:
        init_bot()
    try:
        sessions = bot_instance.loop.session_manager.list_sessions()
        result = []
        for s in sessions:
            created = s["created_at"]
            updated = s["updated_at"]
            if hasattr(created, 'isoformat'):
                created = created.isoformat()
            if hasattr(updated, 'isoformat'):
                updated = updated.isoformat()
            result.append({
                "key": s["key"],
                "created_at": created,
                "updated_at": updated,
                "message_count": s["message_count"]
            })
        result.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"sessions": result}
    except Exception as e:
        return {"sessions": [], "error": str(e)}

@app.put("/sessions/{session_key}")
async def update_session(session_key: str, request: Request):
    if not bot_instance:
        init_bot()
    try:
        body = await request.json()
        title = body.get("title")
        is_pinned = body.get("is_pinned")
        
        session = bot_instance.loop.session_manager.get_or_create(session_key)
        
        if title is not None:
            session.title = title
        if is_pinned is not None:
            session.is_pinned = is_pinned
        
        bot_instance.loop.session_manager.save(session)
        
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/sessions/{session_key}")
async def get_session_detail(session_key: str):
    if not bot_instance:
        init_bot()
    try:
        session = bot_instance.loop.session_manager.get_or_create(session_key)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        messages = []
        if hasattr(session, 'messages') and session.messages:
            for msg in session.messages:
                messages.append({
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", 0)
                })
        
        return {
            "success": True,
            "session": {
                "key": session_key,
                "title": getattr(session, 'title', session_key),
                "messages": messages,
                "is_pinned": getattr(session, 'is_pinned', False)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/sessions/{session_key}")
async def delete_session(session_key: str):
    if not bot_instance:
        init_bot()
    try:
        success = bot_instance.loop.session_manager.delete(session_key)
        return {"success": success}
    except Exception as e:
        return {"success": False, "error": str(e)}
