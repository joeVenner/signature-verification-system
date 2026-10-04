"""Display-only renderings of real pipeline intermediates (no scoring logic here).

Every function takes arrays produced by the verification pipeline itself
(`NormalizedSignature`, `StrokeGeometry`, keypoint coordinates, the RANSAC
transform) and turns them into small PNGs for the live verification console.
Nothing here influences a verdict. All drawing is deterministic: fixed colours,
integer pixel coordinates, OpenCV's deterministic PNG encoder.
"""

from __future__ import annotations

import base64
from typing import Optional, Tuple

import cv2
import numpy as np

from signature_verification_system.src.preprocessing.normalization import canonicalize

MAX_DISPLAY_SIDE = 640
PAPER = 255
# BGR colours: one restrained accent for the reference, charcoal for the questioned.
ACCENT_BGR: Tuple[int, int, int] = (160, 96, 31)       # #1F60A0
INK_BGR: Tuple[int, int, int] = (55, 52, 47)           # #2F3437
FAINT_INK_BGR: Tuple[int, int, int] = (222, 220, 218)
KEYPOINT_BGR: Tuple[int, int, int] = (45, 47, 159)     # #9F2F2D


def encode_png(image: np.ndarray) -> str:
    """Base64 (no data-URI prefix) PNG of a uint8 gray or BGR image.

    Raises:
        ValueError: the image could not be encoded.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def downscale(image: np.ndarray, max_side: int = MAX_DISPLAY_SIDE) -> np.ndarray:
    """Shrink so the longest side is at most `max_side` (never enlarges)."""
    h, w = image.shape[:2]
    factor = max_side / float(max(h, w))
    if factor >= 1.0:
        return image
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def render_original(image: np.ndarray) -> str:
    """The submitted image, downscaled for display only."""
    return encode_png(downscale(_to_bgr(image.astype(np.uint8) if image.dtype != np.uint8 else image)))


def render_harmonised(gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """Background-normalised image with the detected signature bounding box drawn on it."""
    bgr = _to_bgr(gray)
    x, y, w, h = (int(v) for v in bbox)
    thickness = max(2, int(round(max(bgr.shape[:2]) / 300)))
    cv2.rectangle(bgr, (x, y), (x + w - 1, y + h - 1), ACCENT_BGR, thickness)
    return encode_png(downscale(bgr))


def ink_to_gray(ink: np.ndarray) -> np.ndarray:
    """Ink-darkness map (1 = darkest ink) to a uint8 paper-white image."""
    return np.round(255.0 * (1.0 - np.clip(ink, 0.0, 1.0))).astype(np.uint8)


def render_ink_crop(ink: np.ndarray, pad: int = 8) -> str:
    """The extracted, tightly cropped ink map the descriptors are computed on."""
    gray = cv2.copyMakeBorder(ink_to_gray(ink), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=PAPER)
    return encode_png(downscale(gray))


def _plot_points(canvas: np.ndarray, points: np.ndarray, colour: Tuple[int, int, int]) -> None:
    """Paint (x, y) points as 2x2 dots; points outside the canvas are dropped."""
    if points is None or len(points) == 0:
        return
    h, w = canvas.shape[:2]
    xy = np.round(np.asarray(points, dtype=np.float64)).astype(np.int64)
    for dx in (0, 1):
        for dy in (0, 1):
            xs, ys = xy[:, 0] + dx, xy[:, 1] + dy
            keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            canvas[ys[keep], xs[keep]] = colour


def _faint_canvas(ink: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letter-boxed ink map (the keypoint/stroke canvas) drawn as a faint background."""
    canvas = canonicalize(ink, width, height, keep_aspect=True)
    alpha = np.clip(canvas, 0.0, 1.0)[..., None]
    paper = np.full((height, width, 3), PAPER, np.float64)
    faint = np.array(FAINT_INK_BGR, np.float64)
    return np.round(paper * (1.0 - alpha) + faint * alpha).astype(np.uint8)


def render_strokes(ink: np.ndarray, skeleton_points: Optional[np.ndarray], keypoints: Optional[np.ndarray],
                   width: int, height: int) -> str:
    """Skeleton (stroke centre-lines) and SIFT keypoints on the shared comparison canvas."""
    canvas = _faint_canvas(ink, width, height)
    _plot_points(canvas, skeleton_points, INK_BGR)
    if keypoints is not None:
        for x, y in np.round(np.asarray(keypoints, dtype=np.float64)).astype(np.int64):
            cv2.circle(canvas, (int(x), int(y)), 3, KEYPOINT_BGR, 1, cv2.LINE_AA)
    return encode_png(canvas)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to (N, 2) points."""
    m = np.asarray(matrix, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ m[:, :2].T + m[:, 2]


def render_alignment_overlay(reference_points: np.ndarray, questioned_points: np.ndarray,
                             questioned_ink: np.ndarray, matrix: Optional[np.ndarray],
                             width: int, height: int) -> str:
    """Reference skeleton (accent) mapped onto the questioned skeleton (charcoal).

    With `matrix` None the reference is drawn unaligned, in its own canvas position.
    """
    canvas = _faint_canvas(questioned_ink, width, height)
    _plot_points(canvas, questioned_points, INK_BGR)
    moved = reference_points if matrix is None else transform_points(matrix, reference_points)
    _plot_points(canvas, moved, ACCENT_BGR)
    return encode_png(canvas)
