# 便携式部署说明

> 更新日期：2026-07-29

本项目提供多种运行方式，适用于不同使用场景。

## 推荐运行环境

- Windows 10/11 64位
- CPU 推理（默认），也可切换 GPU
- 无需安装 Conda 或 Python（方式一、方式四）

---

## 运行方式总览

| 方式 | 启动文件 | 界面 | 适用场景 |
|------|----------|------|----------|
| Streamlit 网页 | `start_pipeline.bat` | 浏览器 localhost:8501 | 传统使用 |
| FastAPI 后端 | `backend/run_backend.bat` | API + Swagger 文档 | 后端集成 |
| 桌面应用 | `backend/run_desktop.bat` | 原生 Windows 窗口 | 日常使用 |
| EXE 直接运行 | `dist/XRayInspection.exe` | 原生 Windows 窗口 | 无需环境 |
| 微信小程序 | `miniapp/`（开发者工具导入） | 手机端 | 移动端查看 |

---

## 方式一：Streamlit 网页界面

### 1. 拷贝到目标电脑

将整个 `convert` 文件夹拷贝到目标电脑，包含：

| 文件/目录 | 说明 |
|-----------|------|
| `.venv` | 便携式 Python 环境 |
| `f1/` | 文件监控模块 |
| `f2/` | 图像裁剪模块 |
| `f3-yolo/` | YOLO 推理与 Streamlit 界面 |
| `input/` / `output/` / `save/` | 工作目录 |
| `start_pipeline.bat` | **一键启动** |

### 2. 启动

```
1. 双击 start_pipeline.bat
2. 程序自动启动 Streamlit 服务
3. 浏览器自动打开 http://localhost:8501
4. 设置文件夹路径，选择模型权重，点击"开始运行"
```

---

## 方式二：FastAPI 后端

### 启动

```bash
# 使用便携环境
.venv\python.exe backend/main.py

# 或双击
backend\run_backend.bat
```

服务运行在 `http://localhost:8000`，API 文档：`http://localhost:8000/docs`。

### API 调用示例

```bash
curl -X POST http://localhost:8000/api/infer \
  -F "file=@image.dcm" \
  -F "confidence=0.25"
```

### 自动监控

后端启动后自动监控 `C:\Users\PC\Desktop\x光透视识别\DrImage` 目录（可修改 `backend/main.py` 中的 `XRAY_SOURCE` 变量），检测到新的 `0.dcm` 文件自动推理并记录结果。

---

## 方式三：桌面应用

```
1. 双击 backend\run_desktop.bat
2. 等待模型加载（约 1-2 分钟）
3. 原生桌面窗口自动打开
4. 在窗口中上传图片或查看自动监控结果
```

依赖：`pywebview`（已在便携环境中安装）。

---

## 方式四：打包 EXE 直接运行（推荐分发）

```
1. 打开 dist/ 目录
2. 双击 XRayInspection.exe
3. 自动启动后端 + 打开桌面窗口
```

- 无需 Python、无需解压、无需配置
- 模型权重 `best_weights_yolo26x_merged_dataset.pt` 已内置

---

## 方式五：微信小程序

1. 下载并安装 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
2. 导入 `miniapp/` 目录
3. 修改 `utils/` 中的 API 地址为后端服务器地址
4. 预览/上传小程序

> 小程序通过 HTTP 请求调用 FastAPI 后端，需确保后端服务在公网或同一局域网可访问。

---

## 环境依赖

`.venv` 便携 Python 环境包含以下主要依赖：

- Python 3.10
- ultralytics (YOLO)
- PyTorch
- Streamlit
- FastAPI + uvicorn
- OpenCV
- pydicom
- watchdog
- pywebview
- NumPy / Pandas / Pillow / SciPy

---

## 注意事项

- `.venv` 已打包好，直接拷贝使用，**无需在目标机运行 `setup_portable_env.bat`**
- `setup_portable_env.bat` 是开发机重建环境的工具
- `startup_error.log` 记录启动错误信息，排查问题时优先查看
- 端口冲突时：Streamlit 默认 8501，FastAPI 默认 8000，可在对应脚本中修改
- 如杀毒软件拦截，请选择"允许运行"
