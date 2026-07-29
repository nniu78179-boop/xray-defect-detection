import base64
import io
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
F2_DIR = PROJECT_ROOT / "f2"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "save"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".raw", ".dcm"}
POLL_INTERVAL_SECONDS = 1.0
RESULTS_JSON = OUTPUT_DIR / ".results.json"

sys.path.insert(0, str(F2_DIR))
from adaptive_center_crop import (  # noqa: E402
    crop_and_resize,
    detect_core_bbox,
    expand_box,
    load_image,
    parse_ratio,
    save_image,
    to_gray_u8,
)


def load_results_from_disk() -> dict:
    """从磁盘恢复推理结果。"""
    if not RESULTS_JSON.exists():
        return {}
    try:
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    restored = {}
    for key, meta in data.items():
        output_path = Path(key)
        if not output_path.exists():
            continue
        if output_path.stat().st_mtime > meta.get("mtime", 0):
            continue
        annotated_path = output_path.with_name(output_path.stem + "_annotated.jpg")
        if not annotated_path.exists():
            continue
        restored[key] = {
            "path": output_path,
            "mtime": meta["mtime"],
            "confidence": meta["confidence"],
            "original": encode_image_file(output_path),
            "processed": encode_image_file(annotated_path),
            "detections": meta["detections"],
            "inference_time": meta["inference_time"],
            "created_at": meta["created_at"],
        }
    return restored


def save_results_to_disk(results: dict):
    """把当前 session 的推理结果持久化到磁盘。"""
    data = {}
    for key, result in results.items():
        annotated_path = result["path"].with_name(result["path"].stem + "_annotated.jpg")
        save_annotated_jpg(annotated_path, result["processed"])
        data[key] = {
            "mtime": result["mtime"],
            "confidence": result["confidence"],
            "detections": result["detections"],
            "inference_time": result["inference_time"],
            "created_at": result["created_at"],
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def save_annotated_jpg(path: Path, b64_data: str):
    """将 base64 编码的图像存为 JPEG 文件。"""
    import base64 as _b64
    path.write_bytes(_b64.b64decode(b64_data))


def encode_image_file(path: Path) -> str:
    """读取图像文件并编码为 base64。"""
    import base64 as _b64
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return ""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return _b64.b64encode(buffer.getvalue()).decode("utf-8")


st.set_page_config(
    page_title="X光缺陷检测 YOLO 推理系统",
    page_icon="X",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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


def apply_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fa;
            --panel: #ffffff;
            --panel-soft: #eef2f6;
            --text: #16202a;
            --muted: #637083;
            --line: #dbe2ea;
            --teal: #00897b;
            --teal-dark: #00695c;
            --blue: #2374ab;
            --amber: #d68c00;
            --green: #2e9d57;
            --red: #c44536;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden;
        }
        .topbar, .welcome, .panel, .metric-card, .image-panel, .log-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 3px 14px rgba(20, 34, 50, 0.07);
        }
        .topbar {
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .topbar h1, .welcome h1 {
            margin: 0;
            color: var(--text);
            font-size: 1.55rem;
            letter-spacing: 0;
        }
        .topbar p, .welcome p {
            margin: 6px 0 0 0;
            color: var(--muted);
            font-size: 0.92rem;
        }
        .welcome {
            max-width: 860px;
            margin: 8vh auto 0 auto;
            padding: 28px;
        }
        .panel {
            padding: 16px;
            margin-bottom: 16px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }
        .metric-card {
            padding: 16px;
        }
        .metric-card span {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .metric-card strong {
            color: var(--text);
            display: block;
            font-size: 1.85rem;
            line-height: 1.1;
        }
        .image-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .image-panel {
            padding: 14px;
        }
        .image-panel h3, .log-panel h3 {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0 0 12px 0;
        }
        .image-panel img {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 6px;
            max-height: 420px;
            object-fit: contain;
            width: 100%;
        }
        .log-panel {
            padding: 14px;
            margin-top: 14px;
        }
        .log-line {
            align-items: center;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            display: flex;
            gap: 10px;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 0.9rem;
        }
        .log-line:last-child {
            border-bottom: none;
        }
        .status-pill {
            background: rgba(46, 157, 87, 0.12);
            border-radius: 999px;
            color: var(--green);
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 10px;
        }
        .stButton > button {
            background: var(--teal) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover {
            background: var(--teal-dark) !important;
        }
        @media (max-width: 980px) {
            .metrics, .image-grid {
                grid-template-columns: 1fr;
            }
            .welcome {
                margin-top: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "running": False,
        "known_save_files": set(),
        "generated_output_files": set(),
        "results": {},
        "logs": [],
        "last_confidence": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(message: str):
    st.session_state.logs.insert(0, (datetime.now().strftime("%H:%M:%S"), message))
    st.session_state.logs = st.session_state.logs[:80]


def weight_candidates():
    # 从当前目录下的所有文件中筛选出包含"merged"的文件，读取最佳权重
    return sorted(
        p for p in Path(__file__).resolve().parent.glob("*.pt")
        if "merged" in p.name
    )


def normalize_path(path_text: str) -> Path:
    path_text = path_text.strip().strip('"')
    return Path(path_text).expanduser().resolve()


def iter_supported_files(base_dir: Path):
    if not base_dir.exists():
        return []
    return sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def is_file_ready(path: Path) -> bool:
    try:
        size_1 = path.stat().st_size
        time.sleep(0.15)
        size_2 = path.stat().st_size
        return size_1 == size_2
    except OSError:
        return False


def copy_new_save_files(save_dir: Path) -> list[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    current_files = {str(path) for path in iter_supported_files(save_dir)}
    new_paths = [Path(path_text) for path_text in sorted(current_files - st.session_state.known_save_files)]
    copied = []
    pending = set()

    for src in new_paths:
        if not is_file_ready(src):
            pending.add(str(src))
            continue
        rel = src.relative_to(save_dir)
        dst = INPUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
        add_log(f"复制新文件: {src.name} -> input/{rel.as_posix()}")

    st.session_state.known_save_files = current_files - pending
    return copied


def crop_file(input_path: Path) -> Path:
    args = CropArgs()
    image = load_image(input_path, args)
    gray = to_gray_u8(image)
    core_box = detect_core_bbox(
        gray,
        args.min_area_ratio,
        args.keep_x,
        args.keep_y,
        args.bg_threshold_ratio,
    )
    target_ratio = parse_ratio(args.target_ratio)
    crop_box = expand_box(core_box, gray.shape, args.padding, target_ratio)
    cropped = crop_and_resize(image, crop_box, args.resize)

    rel_parent = input_path.parent.relative_to(INPUT_DIR)
    output_dir = OUTPUT_DIR / rel_parent
    output_path = output_dir / f"{input_path.stem}.png"
    save_image(output_path, cropped)
    add_log(f"裁剪完成: {input_path.name} -> {output_path.relative_to(OUTPUT_DIR).as_posix()}")
    return output_path


def crop_new_inputs(input_paths: list[Path]) -> list[Path]:
    outputs = []
    for input_path in input_paths:
        try:
            output_path = crop_file(input_path)
            outputs.append(output_path)
            st.session_state.generated_output_files.add(str(output_path))
        except Exception as exc:
            add_log(f"裁剪失败: {input_path.name} | {exc}")
    return outputs


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str):
    return YOLO(weight_path)


def class_colors(class_labels) -> list[list[int]]:
    base = [
        [0, 0, 255],
        [0, 180, 0],
        [255, 120, 0],
        [180, 0, 180],
        [0, 160, 220],
        [220, 160, 0],
    ]
    count = len(class_labels)
    return [base[index % len(base)] for index in range(count)]


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [255, 0, 0]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 4
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 3),
            0,
            tl / 3,
            [255, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def infer_image(output_path: Path, model, confidence: float):
    import torch
    original = cv2.imdecode(np.fromfile(str(output_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"无法读取图片: {output_path}")

    processed = original.copy()
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

        import torchvision
        keep = torchvision.ops.nms(all_boxes, all_confs, iou_threshold=0.7)

        for idx in keep:
            bbox = all_boxes[idx]
            cnf = all_confs[idx]
            cls_id = int(all_cls[idx])
            class_name = class_labels[cls_id]
            conf_value = float(cnf)
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            plot_one_box(
                [xmin, ymin, xmax, ymax],
                processed,
                label=f"{class_name} {conf_value:.3f}",
                color=colors[cls_id],
                line_thickness=3,
            )
            detections.append([class_name, conf_value])

    inference_time = (time.time() - started) * 1000.0

    return {
        "path": output_path,
        "mtime": output_path.stat().st_mtime,
        "confidence": confidence,
        "original": encode_image(original),
        "processed": encode_image(processed),
        "detections": detections,
        "inference_time": inference_time,
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }


def infer_outputs(output_paths: list[Path], weight_path: str, confidence: float):
    if not output_paths:
        return
    model = load_model(weight_path)
    for output_path in output_paths:
        try:
            result = infer_image(output_path, model, confidence)
            st.session_state.results[str(output_path)] = result
            add_log(f"推理完成: {output_path.name} | {len(result['detections'])} 个目标")
        except Exception as exc:
            add_log(f"推理失败: {output_path.name} | {exc}")
    save_results_to_disk(st.session_state.results)


def outputs_needing_inference(confidence: float) -> list[Path]:
    paths = [
        Path(path_text)
        for path_text in st.session_state.generated_output_files
        if Path(path_text).exists()
    ]
    needs = []
    for path in paths:
        key = str(path)
        result = st.session_state.results.get(key)
        if result is None:
            needs.append(path)
            continue
        if result["mtime"] != path.stat().st_mtime:
            needs.append(path)
            continue
        if abs(float(result["confidence"]) - confidence) > 1e-9:
            needs.append(path)
    return needs


def encode_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_welcome():
    weights = weight_candidates()
    default_weight = weights[0] if weights else Path(__file__).resolve().parent / "best_weights_yolo26x_merged_dataset.pt"
    logo_path = Path(__file__).resolve().parent / "logo.jpg"

    with st.container():
        _, center, _ = st.columns([1, 3, 1])
        with center:
            if logo_path.exists():
                st.image(str(logo_path), width=180)
            st.markdown(
                """
                <div style="margin: 8px 0 20px 0;">
                    <h1 style="margin:0;color:#16202a;font-size:1.55rem;">X光缺陷检测流水线</h1>
                    <p style="margin:6px 0 0 0;color:#637083;font-size:0.92rem;">
                        设置监听目录和 YOLO 权重后，只处理启动后 save 文件夹中新增加的文件。
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            save_dir_text = st.text_input("save 文件夹位置", value=str(DEFAULT_SAVE_DIR))
            weight_options = [str(path) for path in weights]
            selected = st.selectbox(
                "YOLO 权重",
                options=weight_options or [str(default_weight)],
                index=0,
            )
            custom_weight = st.text_input("自定义权重路径", value=selected)
            start = st.button("开始运行", use_container_width=True, type="primary")

    if start:
        save_dir = normalize_path(save_dir_text)
        weight_path = normalize_path(custom_weight)
        if not save_dir.exists():
            st.error(f"save 文件夹不存在: {save_dir}")
            return
        if not weight_path.exists():
            st.error(f"权重文件不存在: {weight_path}")
            return

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.save_dir = str(save_dir)
        st.session_state.weight_path = str(weight_path)
        # 只跳过已有 output 的文件，没有 output 的会在首次轮询时重新处理
        existing = set()
        for sp in iter_supported_files(save_dir):
            rel = sp.relative_to(save_dir)
            if (OUTPUT_DIR / rel.with_suffix(".png")).exists():
                existing.add(str(sp))
        st.session_state.known_save_files = existing
        st.session_state.generated_output_files = set()
        st.session_state.results = load_results_from_disk()
        if st.session_state.results:
            add_log(f"已恢复 {len(st.session_state.results)} 条历史推理结果")
        st.session_state.last_confidence = None
        st.session_state.running = True
        skipped = len(existing)
        total = len(list(iter_supported_files(save_dir)))
        add_log(f"启动监听: {save_dir}")
        add_log(f"已跳过 {skipped} 个已处理文件，{total - skipped} 个待处理")
        add_log(f"加载权重: {weight_path.name}")
        st.rerun()


def render_metrics(results):
    total_images = len(results)
    total_defects = sum(len(item["detections"]) for item in results)
    max_conf = max(
        [det[1] for item in results for det in item["detections"]],
        default=0.0,
    )
    avg_time = (
        sum(float(item["inference_time"]) for item in results) / total_images
        if total_images
        else 0.0
    )
    st.markdown(
        f"""
        <div class="metrics">
            <div class="metric-card"><span>Images</span><strong>{total_images}</strong></div>
            <div class="metric-card"><span>Total Defects</span><strong>{total_defects}</strong></div>
            <div class="metric-card"><span>Max Confidence</span><strong>{max_conf:.2f}</strong></div>
            <div class="metric-card"><span>Avg Time</span><strong>{avg_time:.0f}ms</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    name = result["path"].name
    detections = result["detections"]
    class_fq = dict(Counter(item[0] for item in detections))
    df = pd.DataFrame(class_fq.items(), columns=["类别", "数量"])
    if df.empty:
        df = pd.DataFrame([["未检出", 0]], columns=["类别", "数量"])

    total_defects = len(detections)
    max_conf = max([item[1] for item in detections], default=0.0)

    with st.expander(f"{name} | {total_defects} 个目标 | 最高置信度 {max_conf:.3f}", expanded=True):
        st.markdown(
            f"""
            <div class="image-grid">
                <div class="image-panel">
                    <h3>裁剪图</h3>
                    <img src="data:image/jpeg;base64,{result['original']}" alt="Original">
                </div>
                <div class="image-panel">
                    <h3>推理结果</h3>
                    <img src="data:image/jpeg;base64,{result['processed']}" alt="Prediction">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns([1, 2])
        with left:
            st.dataframe(df, hide_index=True, width="stretch")
        with right:
            rows = [
                {"类别": class_name, "置信度": f"{confidence:.3f}"}
                for class_name, confidence in detections
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_logs():
    st.markdown('<div class="log-panel"><h3>运行日志</h3>', unsafe_allow_html=True)
    if not st.session_state.logs:
        st.markdown('<div class="log-line"><span>等待新文件</span><span></span></div>', unsafe_allow_html=True)
    for log_time, message in st.session_state.logs[:12]:
        st.markdown(
            f'<div class="log-line"><span>{message}</span><span>{log_time}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def run_pipeline_tick(save_dir: Path, weight_path: str, confidence: float):
    copied_inputs = copy_new_save_files(save_dir)
    new_outputs = crop_new_inputs(copied_inputs)

    needs = set(str(path) for path in new_outputs)
    if st.session_state.last_confidence != confidence:
        needs.update(str(path) for path in outputs_needing_inference(confidence))
        st.session_state.last_confidence = confidence
    else:
        needs.update(str(path) for path in outputs_needing_inference(confidence))

    infer_outputs([Path(path) for path in sorted(needs)], weight_path, confidence)


def render_app():
    save_dir = Path(st.session_state.save_dir)
    weight_path = st.session_state.weight_path

    st.markdown(
        f"""
        <div class="topbar">
            <h1>X光缺陷检测 YOLO 推理系统</h1>
            <p><span class="status-pill">运行中</span> 监听: {save_dir} | 权重: {Path(weight_path).name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        confidence = st.slider("置信度阈值", 0.0, 1.0, 0.05, 0.01)
    with col2:
        st.metric("轮询间隔", f"{POLL_INTERVAL_SECONDS:.1f}s")
    with col3:
        stop = st.button("停止运行", use_container_width=True)

    if stop:
        st.session_state.stop_requested = True
        st.rerun()

    run_pipeline_tick(save_dir, weight_path, confidence)

    if st.session_state.get("stop_requested"):
        st.session_state.running = False
        st.session_state.stop_requested = False
        add_log("已停止监听")
        st.rerun()

    result_items = sorted(
        st.session_state.results.values(),
        key=lambda item: item["mtime"],
        reverse=True,
    )
    render_metrics(result_items)

    if result_items:
        for result in result_items[:20]:
            render_result(result)
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:40px 16px;">
                <p style="font-size:1.1rem;color:#637083;margin:0 0 8px 0;">
                    请添加新文件
                </p>
                <p style="font-size:0.85rem;color:#9aa6b5;margin:0;">
                    将 .raw、.dcm 或图片文件放入 save 文件夹，系统会自动检测并推理
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_logs()
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()


apply_styles()
init_state()

if st.session_state.running:
    render_app()
else:
    render_welcome()
