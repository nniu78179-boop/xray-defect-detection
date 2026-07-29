import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".raw", ".dcm"}
RAW_SIZE_PRESETS = {
    # Presets inferred from the current industrial software exports.
    15728640: (2560, 3072, "uint16", 0),
    18874368: (3072, 3072, "uint16", 0),
}
MAGIC_IMAGE_HEADERS = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"BM",
    b"II*\x00",
    b"MM\x00*",
    b"RIFF",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive cropper for preprocessing images before YOLO inference."
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "input"),
        help="Input image directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output"),
        help="Directory to save cropped images.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Extra padding around the detected core box.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.02,
        help="Smallest component area ratio kept during detection.",
    )
    parser.add_argument(
        "--target-ratio",
        type=str,
        default="keep",
        help="Crop aspect ratio, e.g. keep, 1:1, 4:3, 3:4.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional output size. Set 0 to keep the focused crop at native size.",
    )
    parser.add_argument(
        "--keep-x",
        type=float,
        default=0.995,
        help="Horizontal energy ratio kept around the center. Smaller means tighter crop.",
    )
    parser.add_argument(
        "--keep-y",
        type=float,
        default=0.995,
        help="Vertical energy ratio kept around the center. Smaller means tighter crop.",
    )
    parser.add_argument(
        "--bg-threshold-ratio",
        type=float,
        default=0.72,
        help="Threshold ratio above background used to separate bright foreground from dark background.",
    )
    parser.add_argument(
        "--raw-width",
        type=int,
        default=0,
        help="Width for real raw buffers without image headers.",
    )
    parser.add_argument(
        "--raw-height",
        type=int,
        default=0,
        help="Height for real raw buffers without image headers.",
    )
    parser.add_argument(
        "--raw-dtype",
        default="uint16",
        choices=["uint8", "uint16"],
        help="Data type for real raw buffers without image headers.",
    )
    parser.add_argument(
        "--raw-offset",
        type=int,
        default=0,
        help="Bytes to skip before decoding a real raw buffer.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug overlays showing the crop box.",
    )
    return parser.parse_args()


def parse_ratio(ratio_text: str) -> float:
    if ratio_text.lower() == "keep":
        return 0.0
    if ":" in ratio_text:
        width_text, height_text = ratio_text.split(":", 1)
        width = float(width_text)
        height = float(height_text)
        if width <= 0 or height <= 0:
            raise ValueError("Aspect ratio must be positive.")
        return width / height
    ratio = float(ratio_text)
    if ratio <= 0:
        raise ValueError("Aspect ratio must be positive.")
    return ratio


def looks_like_encoded_image(data: bytes) -> bool:
    return any(data.startswith(header) for header in MAGIC_IMAGE_HEADERS)


def load_dicom_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        center = ds.WindowCenter
        width = ds.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        center = float(center)
        width = max(float(width), 1.0)
        low = center - width / 2.0
        high = center + width / 2.0
    else:
        low = float(np.percentile(pixels, 1))
        high = float(np.percentile(pixels, 99))
        if high <= low:
            high = low + 1.0

    pixels = np.clip(pixels, low, high)
    pixels = (pixels - low) / max(high - low, 1e-6)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    return (pixels * 255.0).clip(0, 255).astype(np.uint8)


def infer_raw_preset(path: Path):
    return RAW_SIZE_PRESETS.get(path.stat().st_size)


def normalize_raw_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))
    if high <= low:
        high = low + 1.0
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def load_image(path: Path, args) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        return load_dicom_image(path)

    data = path.read_bytes()

    if looks_like_encoded_image(data):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to decode encoded image: {path}")
        return image

    raw_width = args.raw_width
    raw_height = args.raw_height
    raw_dtype = args.raw_dtype
    raw_offset = args.raw_offset

    if not raw_width or not raw_height:
        preset = infer_raw_preset(path)
        if preset is None:
            raise ValueError(
                f"{path.name} looks like a true raw buffer, but its size does not match any preset. "
                "Please provide --raw-width and --raw-height."
            )
        raw_width, raw_height, raw_dtype, raw_offset = preset

    dtype = np.uint8 if raw_dtype == "uint8" else np.uint16
    flat = np.frombuffer(data, dtype=dtype, offset=raw_offset)
    expected = raw_width * raw_height
    if flat.size < expected:
        raise ValueError(
            f"{path.name} raw data is too short: expected {expected} pixels, got {flat.size}."
        )
    image = flat[:expected].reshape(raw_height, raw_width)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        image = info.max - image
    else:
        image = image.max() - image
    return image


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)
    return normalize_raw_for_display(gray)


def build_candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, dark_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    edge_mask = cv2.Canny(blur, 40, 120)
    edge_mask = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)

    masks = [bright_mask, dark_mask, edge_mask]
    cleaned_masks = []
    kernel_close = np.ones((9, 9), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_masks.append(mask)
    return cleaned_masks


def detect_bright_foreground_bbox(
    gray: np.ndarray,
    padding: float,
    bg_threshold_ratio: float,
) -> tuple[int, int, int, int] | None:
    img_h, img_w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), 7)

    border_band = max(8, min(img_h, img_w) // 24)
    border_pixels = np.concatenate(
        [
            blur[:border_band, :].ravel(),
            blur[-border_band:, :].ravel(),
            blur[:, :border_band].ravel(),
            blur[:, -border_band:].ravel(),
        ]
    )
    bg_level = float(np.percentile(border_pixels, 70))
    fg_level = float(np.percentile(blur, 99.8))
    threshold = bg_level + (fg_level - bg_level) * bg_threshold_ratio
    threshold = min(threshold, fg_level - 1.0) if fg_level > bg_level + 1.0 else bg_level + 1.0

    mask = (blur >= threshold).astype(np.uint8) * 255
    kernel_close = np.ones((19, 19), np.uint8)
    kernel_open = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None

    min_area = max(64, int(img_h * img_w * 0.0003))
    xs: list[int] = []
    ys: list[int] = []
    x2s: list[int] = []
    y2s: list[int] = []

    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        touches_border = x == 0 or y == 0 or x + w >= img_w or y + h >= img_h
        if touches_border:
            continue

        component_mask = labels[y : y + h, x : x + w] == idx
        component_values = blur[y : y + h, x : x + w][component_mask]
        if component_values.size == 0:
            continue
        if float(np.percentile(component_values, 75)) < threshold + 2:
            continue

        xs.append(int(x))
        ys.append(int(y))
        x2s.append(int(x + w))
        y2s.append(int(y + h))

    if not xs:
        return None

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(x2s)
    y2 = max(y2s)

    box_w = x2 - x1
    box_h = y2 - y1
    area_ratio = (box_w * box_h) / float(img_w * img_h)

    # Small isolated parts need more context so their outer edges are not clipped away.
    adaptive_padding = padding
    if area_ratio < 0.03:
        adaptive_padding = max(adaptive_padding, 0.32)
    elif area_ratio < 0.08:
        adaptive_padding = max(adaptive_padding, 0.20)

    pad_x = int(round(box_w * adaptive_padding))
    pad_y = int(round(box_h * adaptive_padding))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def central_mass_bounds(
    projection: np.ndarray,
    keep_ratio: float,
    min_fraction: float,
) -> tuple[int, int]:
    projection = projection.astype(np.float64)
    total = projection.sum()
    length = projection.shape[0]
    if total <= 0:
        margin = max(1, int(length * 0.2))
        return margin, length - margin

    center = int(round(np.dot(np.arange(length), projection) / total))
    target = total * keep_ratio
    left = center
    right = center
    current = projection[center]

    while current < target and (left > 0 or right < length - 1):
        left_value = projection[left - 1] if left > 0 else -1
        right_value = projection[right + 1] if right < length - 1 else -1
        if right_value > left_value:
            right += 1
            current += projection[right]
        else:
            left -= 1
            current += projection[left]

    min_size = max(2, int(length * min_fraction))
    current_size = right - left + 1
    if current_size < min_size:
        pad = (min_size - current_size) // 2 + 1
        left = max(0, left - pad)
        right = min(length - 1, right + pad)

    return left, right + 1


def component_score(
    stats: np.ndarray, centroid: np.ndarray, image_shape: tuple[int, int], min_area: float
) -> float:
    x, y, w, h, area = stats
    if area < min_area:
        return -1.0

    img_h, img_w = image_shape
    cx, cy = centroid
    center_x = img_w / 2.0
    center_y = img_h / 2.0
    dx = abs(cx - center_x) / max(center_x, 1.0)
    dy = abs(cy - center_y) / max(center_y, 1.0)
    center_bonus = max(0.05, 1.4 - 0.9 * dx - 1.1 * dy)

    area_ratio = area / float(img_w * img_h)
    box_fill = area / float(max(w * h, 1))
    slender_penalty = min(w, h) / float(max(w, h, 1))

    return area_ratio * 4.0 + center_bonus * 2.5 + box_fill * 1.2 + slender_penalty * 0.5


def detect_core_bbox(
    gray: np.ndarray,
    min_area_ratio: float,
    keep_x: float,
    keep_y: float,
    bg_threshold_ratio: float,
) -> tuple[int, int, int, int]:
    img_h, img_w = gray.shape
    bright_box = detect_bright_foreground_bbox(gray, padding=0.06, bg_threshold_ratio=bg_threshold_ratio)
    if bright_box is not None:
        return bright_box

    gray_float = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_float, (0, 0), 5)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    if grad.max() > 0:
        grad /= grad.max()

    signal = np.clip(blur - np.quantile(blur, 0.25), 0.0, 1.0)
    energy = signal * 0.75 + grad * 0.25

    y_grid, x_grid = np.mgrid[0:img_h, 0:img_w]
    norm_x = (x_grid - img_w / 2.0) / max(img_w / 2.0, 1.0)
    norm_y = (y_grid - img_h / 2.0) / max(img_h / 2.0, 1.0)
    center_weight = np.exp(-(norm_x**2 * 2.2 + norm_y**2 * 1.5))
    energy = energy * center_weight

    x_proj = energy.sum(axis=0)
    y_proj = energy.sum(axis=1)
    x1, x2 = central_mass_bounds(x_proj, keep_ratio=keep_x, min_fraction=0.30)
    y1, y2 = central_mass_bounds(y_proj, keep_ratio=keep_y, min_fraction=0.20)

    if (x2 - x1) * (y2 - y1) >= img_h * img_w * min_area_ratio:
        return x1, y1, x2, y2

    min_area = img_h * img_w * min_area_ratio
    best_score = -1.0
    best_box = None
    for mask in build_candidate_masks(gray):
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for idx in range(1, count):
            score = component_score(stats[idx], centroids[idx], gray.shape, min_area)
            if score > best_score:
                x, y, w, h, _ = stats[idx]
                best_score = score
                best_box = (int(x), int(y), int(x + w), int(y + h))

    if best_box is not None:
        return best_box

    margin_x = int(img_w * 0.2)
    margin_y = int(img_h * 0.2)
    return margin_x, margin_y, img_w - margin_x, img_h - margin_y


def expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding: float,
    target_ratio: float,
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape
    x1, y1, x2, y2 = box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = box_w * (1.0 + padding * 2.0)
    crop_h = box_h * (1.0 + padding * 2.0)

    if target_ratio > 0:
        current_ratio = crop_w / crop_h
        if current_ratio > target_ratio:
            crop_h = crop_w / target_ratio
        else:
            crop_w = crop_h * target_ratio

    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = int(round(cx + crop_w / 2.0))
    y2 = int(round(cy + crop_h / 2.0))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def crop_and_resize(image: np.ndarray, box: tuple[int, int, int, int], resize: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    if resize and resize > 0:
        cropped = cv2.resize(cropped, (resize, resize), interpolation=cv2.INTER_AREA)
    return cropped


def draw_debug(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if image.ndim == 2:
        overlay = cv2.cvtColor(to_gray_u8(image), cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        overlay = image.copy()

    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return overlay


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_ratio = parse_ratio(args.target_ratio)

    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise SystemExit(f"No supported image files found in {input_dir}")

    print(f"Processing {len(files)} files from: {input_dir}")
    for path in files:
        image = load_image(path, args)
        gray = to_gray_u8(image)
        core_box = detect_core_bbox(
            gray,
            args.min_area_ratio,
            args.keep_x,
            args.keep_y,
            args.bg_threshold_ratio,
        )
        crop_box = expand_box(core_box, gray.shape, args.padding, target_ratio)
        cropped = crop_and_resize(image, crop_box, args.resize)

        output_path = output_dir / f"{path.stem}.png"
        save_image(output_path, cropped)

        print(
            f"{path.name} -> {output_path.name} | src={image.shape[:2]} crop={crop_box} out={cropped.shape[:2]}"
        )

        if args.save_debug:
            debug_path = output_dir / "debug" / f"{path.stem}_debug.png"
            save_image(debug_path, draw_debug(image, crop_box))


if __name__ == "__main__":
    main()
