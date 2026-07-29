# X光缺陷检测系统

基于 YOLO 的 X 光图像缺陷检测流水线，支持**自适应裁剪**预处理与**多端运行**（Streamlit / FastAPI / 桌面应用 / 微信小程序）。

## 核心功能

- **图像格式兼容** — JPG / PNG / BMP / TIFF / WebP / RAW (uint8/uint16) / DICOM (.dcm)
- **自适应裁剪** — 自动识别图像核心有效区域，去除无关背景，提升检测精度
- **YOLO 推理** — TTA + NMS 融合推理，双模型可选
- **文件自动监控** — 监听指定目录，新文件到达即自动处理
- **多端界面** — 网页 / REST API / 原生桌面窗口 / 微信小程序

## 项目结构

```
├── f1/                  # 文件监控模块
├── f2/                  # 自适应图像裁剪核心算法
├── f3-yolo/             # YOLO 推理 + Streamlit 网页界面
├── backend/             # FastAPI 后端 + 桌面应用启动器
├── miniapp/             # 微信小程序前端
├── dist/                # PyInstaller 打包的 Windows EXE
├── input/ output/ save/ # 工作目录
└── start_pipeline.bat   # 一键启动脚本
```

## 快速开始

### 方式一：Streamlit 网页（最简单）

```bash
# Windows 双击
start_pipeline.bat

# 浏览器访问 http://localhost:8501
```

### 方式二：FastAPI 后端

```bash
cd backend
python main.py

# API 文档 http://localhost:8000/docs
```

### 方式三：桌面应用

```bash
cd backend
python desktop_app.py
# 或双击 run_desktop.bat
```

### 方式四：微信小程序

在微信开发者工具中导入 `miniapp/` 目录，配置后端 API 地址。

## API 调用示例

```bash
# 上传图片推理
curl -X POST http://localhost:8000/api/infer \
  -F "file=@image.dcm" \
  -F "confidence=0.25"

# 健康检查
curl http://localhost:8000/api/health

# 最新自动检测结果
curl http://localhost:8000/api/monitor/latest
```

## 自适应裁剪流程

```
原始图片 → 亮度前景检测(Otsu) → 梯度能量分析 → 连通域评分 → 保底裁剪 → 输出
```

## 模型权重

| 权重 | 说明 |
|------|------|
| `best_weights_yolo26x_merged_dataset.pt` | **推荐** — 合并数据集，泛化能力更强 |
| `best_weights_yolo26x_dataset_t.pt` | 特定子集训练 |

## 环境要求

- Windows 10/11 64 位
- Python 3.10+（便携 `.venv` 环境已打包）
- CPU 推理（默认），可选 GPU

## License

MIT
