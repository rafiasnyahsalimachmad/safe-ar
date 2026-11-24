"""Reusable, label-independent feature preprocessing for 21 hand landmarks."""

import numpy as np

LANDMARK_NAMES = (
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
)
FEATURE_COLUMNS = tuple(f'{axis}{index}' for index in range(len(LANDMARK_NAMES)) for axis in 'xyz')
PALM_ANCHORS = (5, 9, 13, 17)
MIN_SCALE_PX = 1e-6


def landmark_array(landmarks):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError('Diperlukan tepat 21 landmark x, y, z yang berhingga.')
    return points


def in_full_image(landmarks, roi, image_size, padding=(0, 0, 0, 0)):
    """Map detector coordinates through input padding and crop to the original image."""
    points = landmark_array(landmarks)
    x0, y0, x1, y1 = roi
    width, height = image_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI di luar gambar atau kosong.')
    left, top, right, bottom = padding
    input_width, input_height = x1 - x0 + left + right, y1 - y0 + top + bottom
    scale = np.array([input_width / width, input_height / height, input_width / width])
    return points * scale + np.array([(x0 - left) / width, (y0 - top) / height, 0.0])


def normalize_landmarks(landmarks, image_size, handedness):
    """Return the normalized 21×3 array and the palm scale in pixel-equivalent units.

    Input is unflipped, full-image MediaPipe coordinates, and handedness is the
    MediaPipe result for that input. Callers can flatten in row-major order.
    """
    points = landmark_array(landmarks)
    width, height = image_size
    if not (np.isfinite([width, height]).all() and width > 0 and height > 0):
        raise ValueError('Dimensi gambar tidak valid.')
    if handedness not in ('Left', 'Right'):
        raise ValueError('Handedness MediaPipe harus Left atau Right.')
    centered = (points - points[0]) * np.array([width, height, width])
    scale = float(np.linalg.norm(centered[list(PALM_ANCHORS)], axis=1).mean())
    if not np.isfinite(scale) or scale <= MIN_SCALE_PX:
        raise ValueError('Skala telapak nol, terlalu kecil, atau tidak berhingga.')
    normalized = centered / scale
    if handedness == 'Left':
        normalized[:, 0] *= -1
    normalized[0] = 0.0
    return normalized, scale
