# miniclaw/server/webhook.py
from fastapi import FastAPI, Request
import asyncio
from pathlib import Path
import json

app = FastAPI()
bot_instance = None

def init_bot():
    from miniclaw.bot import miniclaw
    """初始化 miniclaw 实例"""
    global bot_instance
    if bot_instance is None:
        bot_instance = miniclaw.from_config()

@app.on_event("startup")
async def startup_event():
    """启动时初始化 bot"""
    init_bot()

@app.post("/webhook/msg")
async def msg_webhook(request: Request):
    """接收请求发送的消息"""
    try:
        # 解析请求数据
        data = await request.json()
        message = data.get("message", "")
        session_key = data.get("session_key", "default")
        msg_callback_url = data.get("callback_url")
        
        if not message:
            return {"error": "Message is required"}
        
        # 处理消息
        result = await bot_instance.run(message, session_key)
        
        # 如果提供了回调 URL，将结果发送回请求方
        if msg_callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(msg_callback_url, json={"result": result})
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
