from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
DETECTION_MAX_DIM = 550
GRABCUT_BORDER_RATIO = 0.05
MIN_AREA_RATIO = 0.18


@dataclass(frozen=True)
class QuadDetection:
    corners: np.ndarray
    score: float


def correct_paper(image_bytes: bytes) -> bytes:
    image_format = _detect_format(image_bytes)
    image = _decode_image(image_bytes)

    detection = _detect_with_grabcut(image)
    if detection is None:
        detection = _detect_with_edges(image)

    if detection is None:
        return _encode_image(image, image_format)

    warped = _warp_to_rectangle(image, detection.corners)
    return _encode_image(warped, image_format)


def _detect_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(PNG_MAGIC):
        return ".png"
    if image_bytes.startswith(JPEG_MAGIC):
        return ".jpg"
    return ".png"


def _decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码输入图片。")
    return image


def _encode_image(image: np.ndarray, image_format: str) -> bytes:
    encode_params: list[int] = []
    if image_format == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(image_format, image, encode_params)
    if not ok:
        raise ValueError("无法编码输出图片。")
    return encoded.tobytes()


def _detect_with_grabcut(image: np.ndarray) -> QuadDetection | None:
    small, scale = _resize_for_detection(image)
    height, width = small.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    rect = (
        int(width * GRABCUT_BORDER_RATIO),
        int(height * GRABCUT_BORDER_RATIO),
        int(width * (1.0 - 2 * GRABCUT_BORDER_RATIO)),
        int(height * (1.0 - 2 * GRABCUT_BORDER_RATIO)),
    )
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    return _extract_best_quad(foreground, scale)


def _detect_with_edges(image: np.ndarray) -> QuadDetection | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )
    return _extract_best_quad(edges, scale)


def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_DIM / max(height, width))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _extract_best_quad(mask: np.ndarray, scale: float) -> QuadDetection | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = mask.shape[0] * mask.shape[1]
    best: QuadDetection | None = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_AREA_RATIO:
            continue

        hull = cv2.convexHull(contour)
        quad = _quad_from_hull(hull)
        if quad is None:
            continue

        corners = _order_points(quad / scale)
        score = _score_quad(corners)
        candidate = QuadDetection(corners=corners, score=score)
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def _quad_from_hull(hull: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            return approx[:, 0, :].astype(np.float32)
    if len(hull) >= 4:
        return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return None


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_right = points[np.argmax(sums)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _score_quad(corners: np.ndarray) -> float:
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    area = width_top * height_left
    balance = min(width_top, width_bottom) / max(width_top, width_bottom, 1.0)
    height_balance = min(height_left, height_right) / max(height_left, height_right, 1.0)
    return area * balance * height_balance


def _warp_to_rectangle(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width = max(width, 2)
    height = max(height, 2)

    output_width, output_height = _choose_output_size(image.shape[:2], width, height)
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _choose_output_size(
    original_shape: tuple[int, int],
    detected_width: int,
    detected_height: int,
) -> tuple[int, int]:
    original_height, original_width = original_shape
    ratio = detected_width / max(detected_height, 1)

    if original_width >= original_height:
        output_width = max(detected_width, int(round(original_width * 0.9)))
        output_height = int(round(output_width / ratio))
    else:
        output_height = max(detected_height, int(round(original_height * 0.9)))
        output_width = int(round(output_height * ratio))

    max_side = max(original_width, original_height)
    if max(output_width, output_height) > max_side:
        scale = max_side / max(output_width, output_height)
        output_width = int(round(output_width * scale))
        output_height = int(round(output_height * scale))

    return max(output_width, 2), max(output_height, 2)
