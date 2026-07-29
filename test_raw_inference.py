"""
独立测试脚本：对比增强推理(augment=True) 和 正常推理(augment=False)
不会修改任何原有模块。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "f2"))

import cv2
import numpy as np
from ultralytics import YOLO
from adaptive_center_crop import (
    load_image, to_gray_u8, detect_core_bbox,
    expand_box, crop_and_resize, save_image
)

RAW_PATH = PROJECT_ROOT / "save" / "00025_2026_04_27_17_00_53_614.raw"
WEIGHT_PATH = PROJECT_ROOT / "f3-yolo" / "best_weights_yolo26x_merged_dataset.pt"
CONFIDENCE = 0.05
OUTPUT_DIR = PROJECT_ROOT / "test_debug"


class Args:
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

args = Args()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        print(f"[错误] 文件不存在: {RAW_PATH}")
        return
    if not WEIGHT_PATH.exists():
        print(f"[错误] 权重不存在: {WEIGHT_PATH}")
        return

    print(f"测试文件: {RAW_PATH}")
    print(f"权重文件: {WEIGHT_PATH.name}")
    print(f"置信度阈值: {CONFIDENCE}")
    print("=" * 60)

    # ======== 1. 加载RAW原图 ========
    print("\n[步骤1] 加载RAW原图...")
    raw_image = load_image(RAW_PATH, args)
    print(f"  原始RAW形状: {raw_image.shape}, 数据类型: {raw_image.dtype}")

    if raw_image.ndim == 2:
        img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = raw_image

    # ======== 2. 原图推理 ========
    print("\n[步骤2] 原图推理（未裁剪）...")
    model = YOLO(str(WEIGHT_PATH))

    print("  --- 带 TTA (augment=True) ---")
    results = model.predict(img_bgr, conf=CONFIDENCE, augment=True, verbose=False)
    count_tta = 0
    for r in results:
        count_tta += len(r.boxes)
        for bbox, cnf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            print(f"   类别={model.names[int(cls)]}  置信度={float(cnf):.4f}  框=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]")
    if count_tta == 0:
        print("   未检出任何目标")

    print("  --- 正常推理 (augment=False) ---")
    results = model.predict(img_bgr, conf=CONFIDENCE, augment=False, verbose=False)
    count_normal = 0
    for r in results:
        count_normal += len(r.boxes)
        for bbox, cnf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            print(f"   类别={model.names[int(cls)]}  置信度={float(cnf):.4f}  框=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]")
    if count_normal == 0:
        print("   未检出任何目标")

    # ======== 3. 裁剪后推理 ========
    print("\n[步骤3] 裁剪后推理（模拟f3完整流水线）...")
    gray = to_gray_u8(raw_image)
    core_box = detect_core_bbox(
        gray, args.min_area_ratio, args.keep_x, args.keep_y, args.bg_threshold_ratio,
    )
    print(f"  核心检测框: {core_box}")

    target_ratio = 0.0
    crop_box = expand_box(core_box, gray.shape, args.padding, target_ratio)
    print(f"  最终裁剪框: {crop_box}")

    cropped = crop_and_resize(raw_image, crop_box, args.resize)
    print(f"  裁剪后形状: {cropped.shape}")

    crop_output_path = OUTPUT_DIR / "00025_cropped.png"
    save_image(crop_output_path, cropped)
    print(f"  裁剪图已保存: {crop_output_path}")

    if cropped.ndim == 2:
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    else:
        cropped_bgr = cropped

    print("  --- 裁剪图 + 带 TTA (augment=True) ---")
    results = model.predict(cropped_bgr, conf=CONFIDENCE, augment=True, verbose=False)
    count_crop_tta = 0
    for r in results:
        count_crop_tta += len(r.boxes)
        for bbox, cnf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            print(f"   类别={model.names[int(cls)]}  置信度={float(cnf):.4f}  框=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]")
    if count_crop_tta == 0:
        print("   未检出任何目标")

    print("  --- 裁剪图 + 正常推理 (augment=False) ---")
    results = model.predict(cropped_bgr, conf=CONFIDENCE, augment=False, verbose=False)
    count_crop_normal = 0
    for r in results:
        count_crop_normal += len(r.boxes)
        for bbox, cnf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            print(f"   类别={model.names[int(cls)]}  置信度={float(cnf):.4f}  框=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]")
    if count_crop_normal == 0:
        print("   未检出任何目标")

    print("\n" + "=" * 60)
    print("结果汇总:")
    print(f"  原图 + TTA:          {count_tta} 个目标")
    print(f"  原图 + 正常推理:      {count_normal} 个目标")
    print(f"  裁剪图 + TTA:        {count_crop_tta} 个目标")
    print(f"  裁剪图 + 正常推理:    {count_crop_normal} 个目标")
    print("=" * 60)


if __name__ == "__main__":
    main()
