# start_server.py
from miniclaw.bot import miniclaw
import asyncio

'''
测试用例：
curl -X POST http://localhost:8765/webhook/msg \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，我的名字是张三", "session_key": "user123", "callback_url": "http://localhost:8765/webhook/test"}'
  '''

async def main():
    # 启动 HTTP 服务器
    import threading
    server_thread = threading.Thread(
        target=miniclaw.run_server,
        kwargs={"host": "0.0.0.0", "port": 8765}
    )
    server_thread.daemon = True
    server_thread.start()
    print("HTTP server started on http://0.0.0.0:8765")
    
    # 保持程序运行
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

    