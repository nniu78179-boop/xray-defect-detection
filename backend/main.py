import json
import base64
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torchvision
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# In PyInstaller packaged mode, sys._MEIPASS points to the temp extraction dir.
# Otherwise use the source tree layout.
if getattr(sys, "frozen", False):
    # packaged: resources are next to the exe
    _ROOT = Path(sys.executable).resolve().parent
    _BUNDLE = Path(sys._MEIPASS)
else:
    _ROOT = Path(__file__).resolve().parent.parent
    _BUNDLE = _ROOT

F2_DIR = _BUNDLE / "f2"
sys.path.insert(0, str(F2_DIR))

from adaptive_center_crop import (  # noqa: E402
    crop_and_resize,
    detect_core_bbox,
    expand_box,
    load_image,
    parse_ratio,
    to_gray_u8,
)

# --- config ---
# Look for model next to the exe first, then in the bundle
_MODEL_NAME = "best_weights_yolo26x_merged_dataset.pt"
if (_ROOT / _MODEL_NAME).exists():
    DEFAULT_WEIGHT = str(_ROOT / _MODEL_NAME)
else:
    DEFAULT_WEIGHT = str(_BUNDLE / _MODEL_NAME)
DEFAULT_CONFIDENCE = 0.25
IOU_NMS_THRESHOLD = 0.7

CLASS_NAME_MAP = {
    "defect": "缺陷",
}

# --- auto monitor state ---
AUTO_RESULTS = {}
AUTO_RESULTS_LOCK = threading.Lock()
PROCESSED_FILES = set()
PROCESSED_FILE = Path(__file__).resolve().parent / ".processed.json"
XRAY_SOURCE = r"C:\Users\PC\Desktop\x光透视识别\DrImage"


def _load_processed():
    if PROCESSED_FILE.exists():
        try:
            data = json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
            PROCESSED_FILES.update(data.get("files", []))
        except Exception:
            pass


def _save_processed():
    files = sorted(PROCESSED_FILES)
    if len(files) > 1000:
        files = files[-1000:]
        PROCESSED_FILES.clear()
        PROCESSED_FILES.update(files)
    PROCESSED_FILE.write_text(
        json.dumps({"files": files}, ensure_ascii=False),
        encoding="utf-8",
    )


_load_processed()


class MonitorHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        # only process 0.dcm, ignore _copy, _pre variants
        if path.name != "0.dcm":
            return
        key = str(path.resolve())
        if key in PROCESSED_FILES:
            return  # already processed
        # wait for file to finish writing
        for _ in range(30):
            try:
                s1 = path.stat().st_size
                time.sleep(0.1)
                s2 = path.stat().st_size
                if s1 == s2 and s1 > 0:
                    break
            except OSError:
                time.sleep(0.3)
        else:
            return  # file never stabilized
        self._process(path)

    def _process(self, path: Path):
        key = str(path.resolve())
        try:
            model = getattr(app.state, "model", None)
            if model is None:
                return
            data = path.read_bytes()
            display_name = path.parent.name + ".dcm"
            cropped, _ = crop_image_from_bytes(data, display_name)
            result = infer_image(cropped, model, DEFAULT_CONFIDENCE)

            entry = {
                "filename": display_name,
                "time": datetime.now().strftime("%H:%M:%S"),
                "defect_count": len(result["detections"]),
                "inference_time_ms": round(result["inference_time_ms"], 1),
                "detections": [d.dict() for d in result["detections"]],
                "clean_b64": result["clean_b64"],
                "annotated_b64": result["annotated_b64"],
            }
            PROCESSED_FILES.add(key)
            _save_processed()
            with AUTO_RESULTS_LOCK:
                AUTO_RESULTS[path.name] = entry
                # keep only last 50 results
                keys = list(AUTO_RESULTS.keys())
                for k in keys[:-50]:
                    del AUTO_RESULTS[k]
            print(f"自动检测完成: {path.name} | {len(result['detections'])} 个缺陷")
        except Exception as exc:
            print(f"自动检测失败: {path.name} | {exc}")


class CropArgs:
    padding = 0.0
    min_area_ratio = 0.02
    target_ratio = "keep"
    resize = 0
    keep_x = 0.995
    keep_y = 0.995
    bg_threshold_ratio = 0.72
    raw_width = 0
    raw_height = 0
    raw_dtype = "uint16"
    raw_offset = 0


# --- pydantic models ---
class DetectionItem(BaseModel):
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class InferenceResponse(BaseModel):
    success: bool
    filename: str
    detections: list[DetectionItem]
    defect_count: int
    clean_image_b64: str
    annotated_image_b64: str
    inference_time_ms: float


# --- helpers ---
def class_colors(class_labels) -> list[list[int]]:
    base = [
        [0, 0, 255],
        [0, 180, 0],
        [255, 120, 0],
        [180, 0, 180],
        [0, 160, 220],
        [220, 160, 0],
    ]
    return [base[i % len(base)] for i in range(len(class_labels))]


def _get_chinese_font(size: int) -> ImageFont.FreeTypeFont | None:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return None


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [255, 0, 0]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # draw bounding box with cv2
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        font_size = max(int(tl * 8), 12)
        font = _get_chinese_font(font_size)
        if font is not None:
            # use PIL for Chinese-capable text rendering
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ly = c1[1] - th - 4
            if ly < 0:
                ly = c1[1] + tl
            lx = c1[0]
            # label background
            draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=tuple(color[::-1]))
            # label text in white
            draw.text((lx + 3, ly + 1), label, fill=(255, 255, 255), font=font)
            # convert back to BGR numpy
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # fallback to cv2 (no Chinese support but at least shows something)
            tf = max(tl - 1, 1)
            cv2.rectangle(img, c1, (c1[0] + 200, c1[1] - 20), color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 3), 0, tl / 3,
                        [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def encode_image_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def _is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def crop_image_from_bytes(image_bytes: bytes, filename: str) -> tuple[np.ndarray, str]:
    suffix = Path(filename).suffix.lower()
    # DICOM files must use .dcm suffix or load_image won't recognize them
    if _is_dicom(image_bytes) and suffix != ".dcm":
        suffix = ".dcm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()

        args = CropArgs()
        image = load_image(Path(tmp.name), args)
        gray = to_gray_u8(image)
        core_box = detect_core_bbox(
            gray, args.min_area_ratio, args.keep_x, args.keep_y, args.bg_threshold_ratio,
        )
        target_ratio = parse_ratio(args.target_ratio)
        crop_box = expand_box(core_box, gray.shape, args.padding, target_ratio)
        cropped = crop_and_resize(image, crop_box, args.resize)
    finally:
        os.unlink(tmp.name)

    if len(cropped.shape) == 2:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    log_msg = f"crop {filename}: {image.shape[:2]} -> {cropped.shape[:2]}"
    return cropped, log_msg


def infer_image(image: np.ndarray, model, confidence: float) -> dict:
    processed = image.copy()
    class_labels = model.names
    colors = class_colors(class_labels)

    started = time.time()

    results_tta = model.predict(processed, conf=confidence, augment=True, verbose=False)
    results_normal = model.predict(processed, conf=confidence, augment=False, verbose=False)

    all_boxes, all_confs, all_cls = [], [], []
    for result in results_tta:
        for bbox, cnf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            all_boxes.append(bbox)
            all_confs.append(cnf)
            all_cls.append(cls)
    for result in results_normal:
        for bbox, cnf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            all_boxes.append(bbox)
            all_confs.append(cnf)
            all_cls.append(cls)

    detections = []
    if all_boxes:
        all_boxes = torch.stack(all_boxes)
        all_confs = torch.tensor(all_confs)
        all_cls = torch.tensor(all_cls)
        keep = torchvision.ops.nms(all_boxes, all_confs, iou_threshold=IOU_NMS_THRESHOLD)

        for idx in keep:
            bbox = all_boxes[idx]
            cnf = float(all_confs[idx])
            cls_id = int(all_cls[idx])
            class_name = CLASS_NAME_MAP.get(class_labels[cls_id], class_labels[cls_id])
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            plot_one_box(
                [xmin, ymin, xmax, ymax], processed,
                label=f"{class_name} {cnf:.3f}",
                color=colors[cls_id], line_thickness=3,
            )
            detections.append(DetectionItem(
                class_name=class_name, confidence=cnf,
                x1=xmin, y1=ymin, x2=xmax, y2=ymax,
            ))

    inference_time_ms = (time.time() - started) * 1000.0
    clean_b64 = encode_image_to_b64(image)
    annotated_b64 = encode_image_to_b64(processed)

    return {
        "detections": detections,
        "clean_b64": clean_b64,
        "annotated_b64": annotated_b64,
        "inference_time_ms": inference_time_ms,
    }


# --- FastAPI app ---
app = FastAPI(title="X光缺陷检测 API", version="1.0", description="X光缺陷检测接口，支持图片上传后自动裁剪并推理")

STATIC_DIR = _BUNDLE / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    weight_path = os.environ.get("WEIGHT_PATH", DEFAULT_WEIGHT)
    app.state.model = YOLO(weight_path)
    app.state.weight_name = Path(weight_path).name
    print(f"模型加载完成: {app.state.weight_name}")

    # start folder monitor (recursive: watches date subfolders)
    monitor_path = XRAY_SOURCE if os.path.isdir(XRAY_SOURCE) else str(PROJECT_ROOT / "save")
    os.makedirs(monitor_path, exist_ok=True)
    handler = MonitorHandler()
    observer = Observer()
    observer.schedule(handler, monitor_path, recursive=True)
    observer.start()
    app.state.observer = observer
    print(f"文件夹监控已启动: {monitor_path} (递归)")


@app.on_event("shutdown")
def shutdown():
    obs = getattr(app.state, "observer", None)
    if obs:
        obs.stop()
        obs.join()


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "X光缺陷检测 API", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "weight_file": getattr(app.state, "weight_name", ""),
    }


@app.get("/api/monitor/latest")
def monitor_latest():
    with AUTO_RESULTS_LOCK:
        if not AUTO_RESULTS:
            return {"has_result": False}
        last_key = list(AUTO_RESULTS.keys())[-1]
        entry = AUTO_RESULTS[last_key]
    return {"has_result": True, **entry}


@app.get("/api/monitor/status")
def monitor_status():
    return {
        "monitor_dir": XRAY_SOURCE if os.path.isdir(XRAY_SOURCE) else str(PROJECT_ROOT / "save"),
        "total_processed": len(AUTO_RESULTS),
    }


@app.post("/api/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    display_name: str = Form(""),
):
    contents = await file.read()
    filename = display_name or file.filename or "unknown.png"

    try:
        cropped, crop_log = crop_image_from_bytes(contents, filename)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "filename": filename, "error": f"图片处理失败: {e}"},
        )

    try:
        result = infer_image(cropped, app.state.model, confidence)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "filename": filename, "error": f"推理失败: {e}"},
        )

    return InferenceResponse(
        success=True,
        filename=filename,
        detections=result["detections"],
        defect_count=len(result["detections"]),
        clean_image_b64=result["clean_b64"],
        annotated_image_b64=result["annotated_b64"],
        inference_time_ms=round(result["inference_time_ms"], 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
