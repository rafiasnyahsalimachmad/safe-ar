#!/usr/bin/env python3
"""Extract one annotated hand per image with the MediaPipe Tasks IMAGE API."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing, contextmanager
import csv
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import logging
import math
from multiprocessing import get_context
from pathlib import Path, PurePosixPath
import platform
import random
import sqlite3
import ssl
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.request import urlopen
import zlib

import numpy as np
from PIL import Image, ImageOps
import truststore

from feature_extractor import (
    FEATURE_COLUMNS, LANDMARK_NAMES, MIN_SCALE_PX, PALM_ANCHORS, in_full_image, normalize_landmarks,
)

DATASET = Path(__file__).resolve().parent / 'dataset'
SOURCE = DATASET / 'hagrid_filtered'
OUTPUT = DATASET / 'processed'
CACHE = DATASET / '.landmark_cache'
DEFAULT_MODEL = DATASET.parent / 'models' / 'hand_landmarker.task'
FEATURE_EXPORT_FILES = ('landmarks_raw.csv', 'landmarks_normalized.csv', 'extraction_failures.csv', 'feature_schema.json')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
CLASS_MAP = {'DANGER_STOP': 'palm', 'NEED_MEDICAL': 'fist', 'EVACUATION': 'one', 'SAFE': 'ok', 'NEED_LOGISTICS': 'three'}
REQUIRED = ('image_id', 'image_path', 'safe_ar_label', 'hagrid_label')
SOURCE_METADATA = REQUIRED + (
    'original_split', 'hf_split', 'user_id', 'source_image_path', 'source_size_bytes', 'source_crc32',
    'leading_hand', 'leading_conf',
)
AUDIT = (
    'status', 'width_px', 'height_px', 'target_bbox_index', 'target_bbox_xywh', 'roi_xyxy_px',
    'detector_pass', 'selection_method', 'candidate_count', 'target_bbox_iou', 'target_landmark_fraction',
    'handedness', 'handedness_score', 'detection_confidence', 'palm_scale_px', 'attempts',
)
FEATURE_FIELDS = SOURCE_METADATA + AUDIT + FEATURE_COLUMNS
FAILURE_FIELDS = SOURCE_METADATA + AUDIT + ('failure_reason', 'failure_detail')
NUM_HANDS = 2
MIN_DETECTION = 0.5
MIN_PRESENCE = 0.5
ROI_MARGINS = (('roi', 0.40), ('expanded_roi', 0.75))
MIN_IOU = 0.10
MIN_FRACTION = 0.60
AMBIGUITY_GAP = 0.06
OTHER_BOX_GAP = 0.10
MIN_LEADING_CONF = 0.8
LOG = logging.getLogger('safear.landmarks')
_DETECTOR = None
_ROOT = None


class ImageFailure(ValueError):
    """A specific, auditable reason to exclude a source image."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


@contextmanager
def atomic_file(path, mode='w'):
    options = {'encoding': 'utf-8', 'newline': ''} if mode == 'w' else {}
    with NamedTemporaryFile(mode, dir=path.parent, delete=False, **options) as file:
        temporary = Path(file.name)
        try:
            yield file
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read_metadata(source):
    """Validate the actual header before indexing rows by ID."""
    with (source / 'metadata.csv').open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        missing = set(REQUIRED) - set(fields)
        if missing or len(fields) != len(set(fields)):
            raise ValueError(f'Header metadata.csv tidak dapat digunakan. Kolom wajib yang hilang: {sorted(missing)}.')
        if not {'bboxes', 'labels'}.issubset(fields) and 'annotation_json' not in fields:
            LOG.warning('Header tidak memuat bounding box. Pemilihan akan memakai handedness atau satu kandidat.')
        rows, paths = {}, set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Jumlah kolom metadata tidak valid pada baris {line}.')
            image_id = row['image_id']
            if not image_id or image_id in rows or row['image_path'] in paths:
                raise ValueError(f'ID kosong atau ID/path duplikat pada baris {line}.')
            rows[image_id] = row
            paths.add(row['image_path'])
    if not rows:
        raise ValueError('metadata.csv tidak berisi data.')
    return fields, rows


def source_path(root, row):
    label = row['safe_ar_label']
    if CLASS_MAP.get(label) != row['hagrid_label'] or label not in CLASS_MAP:
        raise ImageFailure('invalid_label', f'Pemetaan tidak valid: {label} / {row["hagrid_label"]}')
    relative = PurePosixPath(row['image_path'])
    if (len(relative.parts) != 2 or relative.parts[0] != label or relative.stem != row['image_id']
            or relative.suffix.lower() not in ('.jpg', '.jpeg') or '\\' in row['image_path']):
        raise ImageFailure('invalid_path', 'Path, folder kelas, atau ID berbeda dari metadata.')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ImageFailure('invalid_path', 'Path gambar berada di luar folder sumber.')
    return path


def hand_annotation(row):
    summary = json.loads(row.get('annotation_json') or '{}')
    if not isinstance(summary, dict):
        raise ValueError('annotation_json harus sebuah objek.')
    pairs = {}
    for key in ('labels', 'bboxes'):
        value = json.loads(row[key]) if row.get(key) else summary.get(key)
        if key in summary and summary[key] != value:
            raise ValueError(f'{key} berbeda dari annotation_json.')
        pairs[key] = value
    boxes, labels = pairs['bboxes'], pairs['labels']
    target = row['hagrid_label']
    lead = str(row.get('leading_hand') or summary.get('leading_hand') or '').lower()
    score = float(row.get('leading_conf') or summary.get('leading_conf') or 0)
    if not math.isfinite(score) or not 0 <= score <= 1 or lead not in {'', 'left', 'right'}:
        raise ValueError('Anotasi tangan utama tidak valid.')
    if labels is not None and (not isinstance(labels, list) or not all(isinstance(label, str) for label in labels)
                               or target not in labels or set(labels) - {target, 'no_gesture'}):
        raise ValueError('Label tangan pada anotasi tidak cocok dengan kelas sumber.')
    if boxes is None or boxes == []:
        return None, [], lead, score
    if not isinstance(boxes, list) or labels is None or len(boxes) != len(labels) or labels.count(target) != 1:
        raise ValueError('Bounding box memerlukan tepat satu pasangan untuk label target.')
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box):
            raise ValueError('Format bounding box harus x, y, lebar, tinggi yang berhingga.')
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError('Bounding box di luar rentang ternormalisasi.')
    return labels.index(target), boxes, lead, score


def decode_image(path, row):
    try:
        expected_size = int(row['source_size_bytes']) if row.get('source_size_bytes') else None
        expected_crc = int(row['source_crc32'], 16) if row.get('source_crc32') else None
    except ValueError as error:
        raise ImageFailure('invalid_source_checksum', str(error)) from error
    try:
        data = path.read_bytes()
        if (expected_size is not None and len(data) != expected_size
                or expected_crc is not None and zlib.crc32(data) != expected_crc):
            raise ImageFailure('source_image_mismatch', 'Ukuran atau CRC berbeda dari ekspor Step 1.')
        with Image.open(BytesIO(data)) as original:
            return original.convert('RGB')
    except FileNotFoundError as error:
        raise ImageFailure('image_missing', 'JPEG sumber tidak ditemukan.') from error
    except (OSError, Image.DecompressionBombError) as error:
        raise ImageFailure('image_read_error', str(error)) from error


def crop_for_box(box, width, height, margin):
    x, y, w, h = box
    dx, dy = w * width, h * height
    cx, cy = x * width + dx / 2, y * height + dy / 2
    radius = max(dx, dy) * (0.5 + margin)
    return max(0, math.floor(cx - radius)), max(0, math.floor(cy - radius)), \
        min(width, math.ceil(cx + radius)), min(height, math.ceil(cy + radius))


def geometry(points, box):
    x, y, w, h = box
    xy = points[:, :2]
    low, high = xy.min(axis=0), xy.max(axis=0)
    box_low = np.array([x, y])
    box_size = np.array([w, h])
    box_high = box_low + box_size
    intersection = np.maximum(0, np.minimum(high, box_high) - np.maximum(low, box_low)).prod()
    iou = float(intersection / (np.prod(high - low) + w * h - intersection))
    pad = 0.15 * box_size
    inside = ((xy >= box_low - pad) & (xy <= box_high + pad)).all(axis=1)
    return iou, float(inside.mean())


def match_score(iou, fraction):
    return 0.7 * iou + 0.3 * fraction


def choose_hand(candidates, target, others, leading, leading_conf):
    if not candidates:
        return None, 'no_hand_detected'
    if target is None:
        matching = [item for item in candidates if item['handedness'].lower() == leading]
        if leading_conf >= MIN_LEADING_CONF and len(matching) == 1:
            return (matching[0], 'annotated_handedness', None, None), None
        if len(candidates) == 1 and (leading_conf < MIN_LEADING_CONF or not leading):
            return (candidates[0], 'single_hand_without_bbox', None, None), None
        return None, 'hand_not_resolved_without_bbox'
    ranked = []
    for candidate in candidates:
        points = candidate['points']
        iou, fraction = geometry(points, target)
        if iou < MIN_IOU or fraction < MIN_FRACTION:
            continue
        score = match_score(iou, fraction)
        other_score = max((match_score(*geometry(points, box)) for box in others), default=0.0)
        if other_score <= score + OTHER_BOX_GAP:
            ranked.append((score, candidate, iou, fraction))
    if not ranked:
        return None, 'no_target_bbox_match'
    ranked.sort(key=lambda item: item[0], reverse=True)
    score, candidate, iou, fraction = ranked[0]
    contenders = [item for item in ranked if score - item[0] < AMBIGUITY_GAP]
    method = 'target_bbox_geometry'
    if len(contenders) > 1:
        matching = [item for item in contenders if item[1]['handedness'].lower() == leading]
        if leading_conf < MIN_LEADING_CONF or len(matching) != 1:
            return None, 'ambiguous_target_hand'
        _, candidate, iou, fraction = matching[0]
        method = 'target_bbox_geometry_handedness_tiebreak'
    return (candidate, method, iou, fraction), None


def init_worker(root, model):
    import mediapipe as mp

    global _DETECTOR, _ROOT
    _ROOT = root
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE, num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION, min_hand_presence_confidence=MIN_PRESENCE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    def detect(image, roi):
        crop = image.crop(roi)
        side = max(crop.size)
        extra_x, extra_y = side - crop.width, side - crop.height
        left, top = extra_x // 2, extra_y // 2
        padding = (left, top, extra_x - left, extra_y - top)
        if extra_x or extra_y:
            crop = ImageOps.expand(crop, border=padding, fill=(0, 0, 0))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop)))
        candidates = []
        for index, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[index][0] if index < len(result.handedness) and result.handedness[index] else None
            points = [(item.x, item.y, item.z) for item in landmarks]
            try:
                points = in_full_image(points, roi, image.size, padding)
            except ValueError:
                continue
            candidates.append({'points': points, 'handedness': (category.category_name or '') if category else '',
                               'handedness_score': category.score if category else None})
        return candidates, len(result.hand_landmarks)

    _DETECTOR = detect


def process_row(row):
    audit = {'status': 'failed'}
    attempts = []
    try:
        path = source_path(_ROOT, row)
        try:
            target_index, boxes, lead, lead_conf = hand_annotation(row)
        except (ValueError, TypeError) as error:
            raise ImageFailure('invalid_annotation', str(error)) from error
        target = boxes[target_index] if target_index is not None else None
        others = [box for index, box in enumerate(boxes) if index != target_index]
        audit.update(target_bbox_index=target_index, target_bbox_xywh=json.dumps(target) if target is not None else '')
        with decode_image(path, row) as image:
            width, height = image.size
            audit.update(width_px=width, height_px=height)
            full = (0, 0, width, height)
            passes = [(name, crop_for_box(target, width, height, margin)) for name, margin in ROI_MARGINS] if target else []
            passes.append(('full_frame', full))
            seen = set()
            for detector_pass, roi in passes:
                if roi in seen:
                    continue
                seen.add(roi)
                attempt = {'pass': detector_pass, 'roi': roi}
                attempts.append(attempt)
                try:
                    candidates, count = _DETECTOR(image, roi)
                except (ValueError, RuntimeError) as error:
                    attempt.update(outcome='mediapipe_error', detail=str(error))
                    continue
                attempt['detected'] = count
                selected, reason = choose_hand(candidates, target, others, lead, lead_conf)
                if not selected:
                    attempt['outcome'] = reason if candidates or not count else 'invalid_mediapipe_landmarks'
                    continue
                candidate, method, iou, contained = selected
                handedness = candidate['handedness']
                score = candidate['handedness_score']
                if handedness not in ('Left', 'Right') or score is None or not 0 <= score <= 1:
                    attempt['outcome'] = 'handedness_unavailable'
                    continue
                try:
                    normalized, scale = normalize_landmarks(candidate['points'], image.size, handedness)
                except ValueError as error:
                    attempt.update(outcome='normalization_failed', detail=str(error))
                    continue
                attempt['outcome'] = 'success'
                audit.update(status='success', detector_pass=detector_pass, roi_xyxy_px=json.dumps(roi),
                             selection_method=method, candidate_count=count, target_bbox_iou=iou,
                             target_landmark_fraction=contained, handedness=handedness,
                             handedness_score=score, palm_scale_px=scale, attempts=json.dumps(attempts))
                return audit, candidate['points'].ravel().tolist(), normalized.ravel().tolist()
        # A geometry rejection or processing issue is more informative than a later miss.
        reason = next((item['outcome'] for item in reversed(attempts) if item['outcome'] != 'no_hand_detected'), 'no_hand_detected')
        raise ImageFailure(reason, 'Tidak ada kandidat yang memenuhi pemeriksaan pada semua percobaan.')
    except ImageFailure as error:
        audit.update(failure_reason=error.reason, failure_detail=str(error), attempts=json.dumps(attempts))
        return audit, None, None


def model_file(path):
    if path.exists():
        if path.is_file() and digest(path) == MODEL_SHA256:
            return path
        raise ValueError(f'SHA-256 file model lokal tidak sesuai: {path}. Pilih path --model yang berbeda.')
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info('Mengunduh model Hand Landmarker FP16 v1 dari Google.')
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urlopen(MODEL_URL, timeout=60, context=context) as response:
        content = response.read(12 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
        raise ValueError('SHA-256 model Hand Landmarker tidak sesuai.')
    with atomic_file(path, 'wb') as file:
        file.write(content)
    return path


def schema(fields, source_digest, selected_count, source_count):
    return {
        'schema_version': 1, 'feature_order': FEATURE_COLUMNS, 'landmark_names': LANDMARK_NAMES,
        'landmark_count': 21, 'feature_count': len(FEATURE_COLUMNS), 'feature_dtype': 'float64',
        'feature_columns_only': 'Gunakan hanya feature_order sebagai vektor input.',
        'raw_coordinates': {
            'x': '(roi_left_px - pad_left_px + x_model * input_width_px) / image_width_px',
            'y': '(roi_top_px - pad_top_px + y_model * input_height_px) / image_height_px',
            'z': 'z_model * input_width_px / image_width_px, kedalaman relatif terhadap pergelangan',
            'input_padding': 'Padding hitam terpusat hingga persegi, tanpa meregangkan crop. Jika ganjil, piksel tambahan di kanan/bawah.',
            'image_convention': 'Orientasi raster JPEG asli, tanpa rotasi EXIF atau mirror.',
            'clipping': 'Nilai keluaran landmark dipertahankan termasuk x/y di luar bingkai.',
        },
        'normalization': {
            'translation': 'Kurangi setiap koordinat dengan koordinat landmark 0 (WRIST).',
            'aspect_ratio': 'Kalikan delta x, y, z dengan (image_width_px, image_height_px, image_width_px).',
            'scale': 'Rata-rata jarak Euclidean 3D dari WRIST ke empat MCP, dalam satuan ekuivalen piksel.',
            'scale_landmarks': PALM_ANCHORS, 'minimum_scale_px_exclusive': MIN_SCALE_PX,
            'handedness': 'Jika MediaPipe mengembalikan Left, kalikan x dengan -1. Right dipertahankan. Handedness yang hilang → gagal.',
            'rotation': 'Orientasi bidang gambar dipertahankan.',
            'units': 'Nirunit, relatif terhadap ukuran telapak.',
        },
        'mediapipe': {
            'version': version('mediapipe'), 'api': 'MediaPipe Tasks HandLandmarker', 'mode': 'IMAGE', 'delegate': 'CPU',
            'model_url': MODEL_URL, 'model_sha256': MODEL_SHA256, 'num_hands': NUM_HANDS,
            'min_hand_detection_confidence': MIN_DETECTION, 'min_hand_presence_confidence': MIN_PRESENCE,
            'handedness_score': 'Skor klasifikasi Left/Right dari API.',
            'detection_confidence': 'Kosong karena API hasil tidak mengembalikan skor deteksi telapak atau presence.',
        },
        'selection': {
            'target_bbox': 'Pasangan bboxes/labels yang sesuai dengan hagrid_label, dipakai untuk memilih tangan.',
            'retry_order': [name for name, _ in ROI_MARGINS] + ['full_frame'],
            'roi_margin_per_side': dict(ROI_MARGINS), 'roi_units': 'kotak persegi dalam koordinat piksel, dipotong pada batas gambar',
            'min_target_iou': MIN_IOU, 'min_landmark_fraction_in_target': MIN_FRACTION,
            'target_padding_for_fraction': 0.15, 'geometry_score': '0.7 * IoU + 0.3 * fraksi landmark',
            'minimum_ambiguity_gap': AMBIGUITY_GAP, 'competing_annotation_score_gap': OTHER_BOX_GAP,
            'min_leading_conf_for_tiebreak': MIN_LEADING_CONF,
            'missing_bbox': 'Full frame: gunakan handedness yang sesuai anotasi tepercaya atau satu kandidat tanpa petunjuk handedness tepercaya.',
        },
        'provenance': {
            'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {name: version(name) for name in ('numpy', 'Pillow', 'opencv-contrib-python')},
            'source_header': fields, 'source_metadata_sha256': source_digest, 'source_rows': source_count,
            'selected_rows': selected_count, 'image_path_root': 'hagrid_filtered/', 'class_mapping': CLASS_MAP,
        },
        'feature_metadata_columns': SOURCE_METADATA + AUDIT, 'failure_columns': FAILURE_FIELDS,
    }


def select_rows(rows, sample_per_class):
    if not sample_per_class:
        return list(rows.values())
    rng = random.Random(42)
    selected = set()
    for label in CLASS_MAP:
        pool = [image_id for image_id, row in rows.items() if row['safe_ar_label'] == label]
        selected.update(rng.sample(pool, min(len(pool), sample_per_class)))
    return [row for key, row in rows.items() if key in selected]


def checkpoint_path(cache, source, output, spec):
    files = (Path(__file__).resolve(), Path(__file__).with_name('feature_extractor.py').resolve())
    identity = [str(source), str(output), spec, [digest(file) for file in files]]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cache / f'{key[:24]}.sqlite3'


def export(output, selected, db, spec):
    outputs = [('landmarks_raw.csv', FEATURE_FIELDS), ('landmarks_normalized.csv', FEATURE_FIELDS),
               ('extraction_failures.csv', FAILURE_FIELDS)]
    stats = Counter()
    with ExitStack() as stack:
        writers = [csv.DictWriter(stack.enter_context(atomic_file(output / name)), fieldnames=fields) for name, fields in outputs]
        for writer in writers:
            writer.writeheader()
        raw_csv, norm_csv, failure_csv = writers
        for row in selected:
            image_id = row['image_id']
            payload, = db.execute('SELECT payload FROM results WHERE image_id=?', (image_id,)).fetchone()
            audit, raw, normalized = json.loads(payload)
            base = {key: row.get(key, '') for key in SOURCE_METADATA} | audit
            stats[audit['status']] += 1
            if audit['status'] == 'success':
                raw_csv.writerow(base | dict(zip(FEATURE_COLUMNS, raw, strict=True)))
                norm_csv.writerow(base | dict(zip(FEATURE_COLUMNS, normalized, strict=True)))
            else:
                failure_csv.writerow(base)
    with atomic_file(output / 'feature_schema.json') as file:
        json.dump(spec, file, ensure_ascii=False, indent=2)
        file.write('\n')
    return stats


def run(args):
    source, output, cache, model_path = (path.resolve() for path in (args.source, args.output, args.cache, args.model))
    if any(path.is_relative_to(source) for path in (output, cache, model_path)):
        raise ValueError('Output, cache, dan model harus berada di luar folder gambar sumber.')
    existing_exports = [name for name in FEATURE_EXPORT_FILES if (output / name).exists()]
    if existing_exports:
        if len(existing_exports) == len(FEATURE_EXPORT_FILES):
            LOG.info('Ekspor sudah ada di %s. Gunakan validator, atau pilih --output yang baru.', output)
            return
        raise ValueError(f'Ekspor lama di {output} tidak lengkap. Pilih --output baru untuk mempertahankan file tersebut.')
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    fields, rows = read_metadata(source)
    selected = select_rows(rows, args.sample_per_class)
    LOG.info('Schema sumber: %s. Memilih %d dari %d gambar.', fields, len(selected), len(rows))
    spec = schema(fields, digest(source / 'metadata.csv'), len(selected), len(rows))
    spec['provenance']['sample_per_class'] = args.sample_per_class
    model = model_file(model_path)
    with closing(sqlite3.connect(checkpoint_path(cache, source, output, spec))) as db:
        db.execute('CREATE TABLE IF NOT EXISTS results (image_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        existing = {image_id for image_id, in db.execute('SELECT image_id FROM results')}
        pending = [row for row in selected if row['image_id'] not in existing]
        LOG.info('%d hasil cache, %d gambar menunggu, %d worker.', len(selected) - len(pending), len(pending), args.workers)
        if pending:
            start = time.monotonic()
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'),
                                     initializer=init_worker, initargs=(source, model)) as pool:
                for index, (row, result) in enumerate(zip(pending, pool.map(process_row, pending, chunksize=12), strict=True), 1):
                    db.execute('INSERT OR REPLACE INTO results VALUES (?, ?)', (row['image_id'], json.dumps(result, allow_nan=False)))
                    if index % 200 == 0:
                        db.commit()
                    if index % 500 == 0 or index == len(pending):
                        LOG.info('Diproses %d/%d hasil baru (%.1f gambar/detik).', index, len(pending), index / (time.monotonic() - start))
            db.commit()
        stats = export(output, selected, db, spec)
    LOG.info('Ekspor selesai di %s. %d berhasil, %d gagal.', output, stats['success'], stats['failed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Path model FP16 v1. Unduh otomatis jika belum ada.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-per-class', type=int, default=0, help='Jumlah acak per kelas untuk uji (0 = semua gambar).')
    args = parser.parse_args()
    if not (1 <= args.workers <= 16) or args.sample_per_class < 0:
        parser.error('--workers harus 1–16 dan --sample-per-class minimal 0.')
    if not ((3, 11) <= sys.version_info[:2] <= (3, 12)):
        parser.error('Gunakan Python 3.11 atau 3.12 dengan dependensi di requirements.txt.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        LOG.error('Ekstraksi terhenti: %s', error)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
