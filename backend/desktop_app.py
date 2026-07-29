"""X光缺陷检测 — 桌面应用启动器

启动 FastAPI 后端并打开原生桌面窗口。
复用 static/index.html 作为界面。

兼容开发模式和 PyInstaller 打包模式。
"""

import sys
import threading
import time
from pathlib import Path

import uvicorn
import webview

if getattr(sys, "frozen", False):
    # PyInstaller packaged: code is at the root of the bundle
    sys.path.insert(0, str(Path(sys._MEIPASS)))
else:
    # Development: add backend dir to path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app  # noqa: E402


def main():
    host = "127.0.0.1"
    port = 8000

    # 在后台线程启动 uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # 等待服务就绪（模型加载需要时间，最长等 2 分钟）
    import json, urllib.request
    ready = False
    for i in range(240):
        try:
            r = urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=1)
            if r.status == 200:
                data = json.loads(r.read())
                if data.get("model_loaded"):
                    ready = True
                    break
        except Exception:
            pass
        if i == 0:
            print("正在加载 YOLO 模型，请稍候...")
        time.sleep(0.5)

    if not ready:
        print("[错误] 模型加载超时，请检查控制台输出")
        sys.exit(1)

    print("模型加载完成，启动桌面窗口...")

    url = f"http://{host}:{port}"

    # 打开桌面窗口
    webview.create_window(
        title="X光缺陷检测系统",
        url=url,
        width=1100,
        height=800,
        min_size=(800, 600),
        text_select=True,
    )
    webview.start()

    # 窗口关闭后退出
    server.should_exit = True
    print("应用已退出")


if __name__ == "__main__":
    main()
