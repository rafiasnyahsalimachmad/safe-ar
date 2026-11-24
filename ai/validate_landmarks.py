#!/usr/bin/env python3
"""Validate the landmark export against Step 1 and render annotated visual samples."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
import json
import logging
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from extract_landmarks import (
    CLASS_MAP, FAILURE_FIELDS, FEATURE_FIELDS, MIN_FRACTION, MIN_IOU, NUM_HANDS, OUTPUT, SOURCE,
    SOURCE_METADATA, atomic_file, crop_for_box, digest, geometry, hand_annotation, read_metadata, select_rows,
)
from feature_extractor import FEATURE_COLUMNS, LANDMARK_NAMES, normalize_landmarks

LOG = logging.getLogger('safear.landmarks.validation')
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def csv_rows(path, fields):
    with path.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f'Header {path.name} tidak cocok dengan skema.')
        yield from reader


def check_source(row, expected, status):
    if None in row or any(value is None for value in row.values()):
        raise ValueError('Jumlah kolom pada baris tidak valid.')
    source = expected.get(row['image_id'])
    if source is None:
        raise ValueError('ID hasil tidak tercatat pada manifest yang dipilih.')
    if row['status'] != status or any(row[key] != source.get(key, '') for key in SOURCE_METADATA):
        raise ValueError('Status atau metadata identitas tidak cocok dengan manifest.')
    return source


def check_features(raw, normalized, source):
    raw_points, normalized_points = (
        np.array([float(row[key]) for key in FEATURE_COLUMNS], dtype=np.float64).reshape(21, 3)
        for row in (raw, normalized)
    )
    if not np.isfinite(raw_points).all() or not np.isfinite(normalized_points).all():
        raise ValueError('Fitur memuat NaN atau infinity.')
    width, height = int(raw['width_px']), int(raw['height_px'])
    recomputed, scale = normalize_landmarks(raw_points, (width, height), raw['handedness'])
    if np.any(normalized_points[0] != 0) or not np.allclose(normalized_points, recomputed, rtol=1e-10, atol=1e-12):
        raise ValueError('Fitur normalisasi berbeda dari perhitungan ulang.')
    if not math.isclose(scale, float(raw['palm_scale_px']), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Skala audit berbeda dari perhitungan ulang.')
    x0, y0, x1, y1 = json.loads(raw['roi_xyxy_px'])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI berada di luar raster gambar.')
    attempts = json.loads(raw['attempts'])
    count = int(raw['candidate_count'])
    if (not attempts or attempts[-1]['outcome'] != 'success' or attempts[-1]['pass'] != raw['detector_pass']
            or attempts[-1]['roi'] != [x0, y0, x1, y1] or attempts[-1]['detected'] != count or not 1 <= count <= NUM_HANDS):
        raise ValueError('Log percobaan atau jumlah kandidat tidak konsisten.')
    if raw['detection_confidence'] or not 0.0 <= float(raw['handedness_score']) <= 1.0:
        raise ValueError('Skor handedness atau kolom confidence deteksi tidak valid.')
    index, boxes, _, _ = hand_annotation(source)
    method = raw['selection_method']
    if index is None:
        if method not in {'annotated_handedness', 'single_hand_without_bbox'} or raw['target_bbox_xywh']:
            raise ValueError('Metode pemilihan tanpa bounding box tidak valid.')
    else:
        if index != int(raw['target_bbox_index']) or boxes[index] != json.loads(raw['target_bbox_xywh']):
            raise ValueError('Bounding box target berbeda dari pasangan label/kotak sumber.')
        iou, fraction = geometry(raw_points, boxes[index])
        if (method not in {'target_bbox_geometry', 'target_bbox_geometry_handedness_tiebreak'}
                or iou < MIN_IOU or fraction < MIN_FRACTION
                or not math.isclose(iou, float(raw['target_bbox_iou']), abs_tol=1e-10)
                or not math.isclose(fraction, float(raw['target_landmark_fraction']), abs_tol=1e-10)):
            raise ValueError('Landmark tidak memenuhi kecocokan kotak target.')
    return raw_points, boxes, index


class VisualSelection:
    def __init__(self):
        self.random = random.Random(73)
        self.typical = defaultdict(list)
        self.two_boxes = {}
        self.fallback = {}

    def add(self, label, row, points, boxes, target_index):
        item = (row, points, boxes, target_index)
        keys = self.typical[label]
        keys.append((self.random.random(), item))
        keys.sort(key=lambda pair: pair[0])
        del keys[4:]
        if len(boxes) > 1 and label not in self.two_boxes:
            self.two_boxes[label] = item
        if row['detector_pass'] != 'roi' and label not in self.fallback:
            self.fallback[label] = item

    def rows(self, label):
        candidates = [self.two_boxes.get(label), self.fallback.get(label)] + [item for _, item in self.typical[label]]
        unique = {}
        for item in candidates:
            if item is not None:
                unique.setdefault(item[0]['image_id'], item)
        return list(unique.values())[:4]


def validate(source_dir, output, spec):
    fields, rows = read_metadata(source_dir)
    provenance = spec['provenance']
    if (tuple(spec['feature_order']) != FEATURE_COLUMNS or tuple(spec['landmark_names']) != LANDMARK_NAMES
            or spec['feature_count'] != 63 or spec['landmark_count'] != 21
            or provenance['class_mapping'] != CLASS_MAP or provenance['source_header'] != fields
            or provenance['source_rows'] != len(rows) or provenance['source_metadata_sha256'] != digest(source_dir / 'metadata.csv')):
        raise ValueError('Feature schema berbeda dari kontrak fitur atau metadata Step 1.')
    selected = {row['image_id']: row for row in select_rows(rows, provenance['sample_per_class'])}
    if len(selected) != provenance['selected_rows']:
        raise ValueError('Jumlah baris yang dipilih berbeda dari skema.')
    total_by_class = Counter(row['safe_ar_label'] for row in selected.values())
    seen, successes, failures = set(), Counter(), Counter()
    reasons, passes, methods, hands = Counter(), Counter(), Counter(), Counter()
    per_class_reasons = defaultdict(Counter)
    two_hands, two_boxes, oob, lead_count, lead_match = 0, 0, 0, 0, 0
    errors, error_examples = Counter(), []
    visual = VisualSelection()

    def issue(category, detail):
        errors[category] += 1
        if len(error_examples) < 30:
            error_examples.append(f'{category} | {detail}')

    def mark(image_id):
        if image_id in seen:
            raise ValueError('ID hasil duplikat atau muncul pada CSV berhasil dan gagal.')
        seen.add(image_id)

    raw_csv = csv_rows(output / 'landmarks_raw.csv', FEATURE_FIELDS)
    norm_csv = csv_rows(output / 'landmarks_normalized.csv', FEATURE_FIELDS)
    for line, pair in enumerate(zip_longest(raw_csv, norm_csv), 2):
        raw, normalized = pair
        if raw is None or normalized is None:
            issue('paired_csv_length', f'Baris {line}')
            continue
        try:
            base = check_source(raw, selected, 'success')
            check_source(normalized, selected, 'success')
            if any(raw[key] != normalized[key] for key in FEATURE_FIELDS[:-len(FEATURE_COLUMNS)]):
                raise ValueError('Metadata raw dan normalized berbeda.')
            mark(raw['image_id'])
            points, boxes, target = check_features(raw, normalized, base)
            label = raw['safe_ar_label']
            successes[label] += 1
            passes[raw['detector_pass']] += 1
            methods[raw['selection_method']] += 1
            hands[raw['handedness']] += 1
            two_hands += int(raw['candidate_count']) > 1
            two_boxes += len(boxes) > 1
            oob += bool(((points[:, :2] < 0) | (points[:, :2] > 1)).any())
            if raw['leading_hand']:
                lead_count += 1
                lead_match += raw['leading_hand'].lower() == raw['handedness'].lower()
            visual.add(label, raw, points, boxes, target)
        except (KeyError, TypeError, ValueError) as error:
            issue('invalid_features', f'Baris {line}, ID {raw["image_id"]} | {error}')
        if line % 5000 == 0:
            LOG.info('Memvalidasi CSV fitur hingga baris %d.', line)

    for line, row in enumerate(csv_rows(output / 'extraction_failures.csv', FAILURE_FIELDS), 2):
        try:
            check_source(row, selected, 'failed')
            mark(row['image_id'])
            reason = row['failure_reason']
            attempts = json.loads(row['attempts'])
            if not reason or not row['failure_detail'] or not isinstance(attempts, list) or any(
                not isinstance(item, dict) or not item.get('outcome') or item['outcome'] == 'success' for item in attempts
            ):
                raise ValueError('Alasan atau percobaan gagal tidak valid.')
            label = row['safe_ar_label']
            failures[label] += 1
            reasons[reason] += 1
            per_class_reasons[reason][label] += 1
        except (KeyError, TypeError, ValueError) as error:
            issue('invalid_failure', f'Baris {line}, ID {row["image_id"]} | {error}')
    for image_id in sorted(selected.keys() - seen):
        issue('missing_result', image_id)
    for label, expected in total_by_class.items():
        if successes[label] + failures[label] != expected:
            issue('class_count_mismatch', label)
        if label in CLASS_MAP and not successes[label]:
            issue('empty_successful_class', label)
    stats = {
        'total': total_by_class, 'success': successes, 'failure': failures, 'reasons': reasons,
        'per_class_reasons': per_class_reasons, 'passes': passes, 'methods': methods, 'handedness': hands,
        'two_candidates': two_hands, 'two_boxes': two_boxes, 'out_of_frame': oob, 'lead_count': lead_count,
        'lead_match': lead_match, 'errors': errors, 'examples': error_examples,
    }
    return stats, visual


def preview_transform(image, region, panel):
    origin_x, origin_y, max_width, max_height = panel
    displayed = ImageOps.contain(image.crop(region), (max_width, max_height))
    offset_x = origin_x + (max_width - displayed.width) // 2
    offset_y = origin_y + (max_height - displayed.height) // 2
    fx = displayed.width / (region[2] - region[0])
    fy = displayed.height / (region[3] - region[1])

    def map_xy(px, py):
        return offset_x + (px - region[0]) * fx, offset_y + (py - region[1]) * fy

    return displayed, (offset_x, offset_y), map_xy


def draw_preview(sheet, image, region, panel, boxes, target, points):
    preview, position, to_display = preview_transform(image, region, panel)
    sheet.paste(preview, position)
    draw = ImageDraw.Draw(sheet)
    width, height = image.size
    for index, (x, y, w, h) in enumerate(boxes):
        left = max(region[0], x * width)
        top = max(region[1], y * height)
        right = min(region[2], (x + w) * width)
        bottom = min(region[3], (y + h) * height)
        if left < right and top < bottom:
            color = '#ffd34d' if index == target else '#a9b5c3'
            draw.rectangle((to_display(left, top), to_display(right, bottom)), outline=color, width=2)
    mapped = [to_display(x * width, y * height) for x, y, _ in points]
    bounds = (position[0], position[1], position[0] + preview.width, position[1] + preview.height)

    def visible(point):
        return bounds[0] <= point[0] < bounds[2] and bounds[1] <= point[1] < bounds[3]

    for a, b in CONNECTIONS:
        if visible(mapped[a]) and visible(mapped[b]):
            draw.line((mapped[a], mapped[b]), fill='#45e7ff', width=2)
    for index, (x, y) in enumerate(mapped):
        if visible((x, y)):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='#ffffff' if index else '#fb5581')


def visual_samples(source_dir, output, selection):
    destination = output / 'landmark_visual_samples'
    destination.mkdir(exist_ok=True)
    written = 0
    heading, caption = ImageFont.load_default(19), ImageFont.load_default(13)
    for label, hagrid in CLASS_MAP.items():
        examples = selection.rows(label)
        if not examples:
            continue
        card_width, gutter = 323, 10
        sheet = Image.new('RGB', (len(examples) * (card_width + gutter) + gutter, 323), '#111827')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f'{label} / {hagrid}', fill='#edf5ff', font=heading)
        draw.text((10, 32), 'Kuning = kotak target  |  Abu-abu = tangan lain  |  Biru = landmark MediaPipe', fill='#ced8e4', font=caption)
        for col, (row, points, boxes, target) in enumerate(examples):
            x = gutter + col * (card_width + gutter)
            draw.rounded_rectangle((x, 55, x + card_width, 310), radius=5, fill='#24324a')
            with Image.open(source_dir / row['image_path']) as original:
                image = original.convert('RGB')
                full = (0, 0, image.width, image.height)
                region = crop_for_box(boxes[target], *image.size, 0.28) if target is not None else full
                draw_preview(sheet, image, full, (x + 5, 60, 147, 200), boxes, target, points)
                draw_preview(sheet, image, region, (x + 156, 60, 160, 200), boxes, target, points)
            draw.text((x + 8, 266), row['image_id'], fill='#edf5ff', font=caption)
            gap = f" | IoU {float(row['target_bbox_iou']):.2f}" if row['target_bbox_iou'] else ''
            draw.text((x + 8, 285), f"{row['detector_pass']} | {row['handedness']}{gap}", fill='#b7d1f1', font=caption)
        with atomic_file(destination / f'{label}.jpg', 'wb') as file:
            sheet.save(file, format='JPEG', quality=92)
        written += len(examples)
    return written


def percent(numerator, denominator):
    return f'{100 * numerator / denominator:.2f}%' if denominator else '—'


def report(output, spec, stats, visual_count):
    total = sum(stats['total'].values())
    ok = sum(stats['success'].values())
    failed = sum(stats['failure'].values())
    state = 'gagal' if stats['errors'] else 'lulus'
    lines = [
        '# S.A.F.E. AR Kelompok 6 — Step 2', '',
        f'**Validasi dataset fitur {state}.** {ok:,} dari {total:,} gambar berhasil ({percent(ok, total)}). '
        f'{failed:,} gagal diekspor sebagai fitur dan dicatat di [extraction_failures.csv](extraction_failures.csv).', '',
        '## Jumlah dan distribusi kelas', '',
        '| S.A.F.E. AR | HaGRID | Diproses | Berhasil | Gagal | Tingkat berhasil | Porsi fitur akhir |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for label, hagrid in CLASS_MAP.items():
        n, success, failure = (stats[key][label] for key in ('total', 'success', 'failure'))
        lines.append(f'| {label} | `{hagrid}` | {n:,} | {success:,} | {failure:,} | {percent(success, n)} | {percent(success, ok)} |')
    lines.extend([
        f'| **Total** | | **{total:,}** | **{ok:,}** | **{failed:,}** | **{percent(ok, total)}** | **{percent(ok, ok)}** |',
        '', '## Sumber dan detektor', '',
        f'- Manifest menggunakan {len(spec["provenance"]["source_header"])} kolom dari Step 1 dan memilih {spec["provenance"]["selected_rows"]:,} dari {spec["provenance"]["source_rows"]:,} gambar.',
        '- Setiap baris membawa `image_id`, `image_path` relatif terhadap `hagrid_filtered/`, kedua label, `original_split`, `hf_split`, dan `user_id` dari sumber. `original_split=train_val` mencatat split gabungan di ZIP.',
        f'- Python {spec["provenance"]["python"]}, MediaPipe {spec["mediapipe"]["version"]}, Hand Landmarker FP16 v1 dalam mode IMAGE pada CPU. [Dokumentasi MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python)',
        '- Detektor mencoba crop persegi di sekitar bounding box gestur, crop dengan margin lebih besar, dan gambar penuh. Padding hitam dihapus secara matematis saat koordinat dikembalikan ke raster sumber.',
        '- Kandidat dipilih berdasarkan IoU dan fraksi landmark pada kotak gestur. Kotak tangan lain diperhitungkan. Handedness anotasi dapat memutus seri jika `leading_conf` memenuhi ambang. Parameter dan ambang tercatat di [feature_schema.json](feature_schema.json).',
        '- `handedness_score` menyimpan keyakinan klasifikasi Left/Right. `detection_confidence` kosong karena hasil API gambar hanya mengekspos landmark, world landmarks, dan klasifikasi handedness. Ambang deteksi tersimpan dalam skema.',
        '', '## Preprocessing fitur', '',
        '- [landmarks_raw.csv](landmarks_raw.csv) menyimpan x dan y sebagai fraksi lebar dan tinggi gambar asli. z merupakan estimasi kedalaman relatif pergelangan yang dinyatakan dalam satuan lebar gambar.',
        '- [landmarks_normalized.csv](landmarks_normalized.csv) menyimpan `x0, y0, z0, …, x20, y20, z20`. Translasi memakai pergelangan (0). Delta dikalikan dengan `(W, H, W)` lalu dibagi rata-rata jarak 3D dari pergelangan ke MCP (5, 9, 13, 17).',
        '- Untuk tangan `Left`, komponen x dicerminkan menuju konvensi tangan kanan berdasarkan hasil MediaPipe. Label gestur hanya dipakai untuk memilih kotak anotasi. Vektor fitur hanya berisi koordinat.',
        '', '## Pemeriksaan kualitas', '',
        f'- {ok:,} pasangan raw/normalized lolos pemeriksaan 63 nilai berhingga, urutan fitur, normalisasi ulang, metadata dan sumber bounding box. Terdapat {sum(stats["errors"].values()):,} temuan validasi.',
        f'- {stats["two_boxes"]:,} gambar berhasil punya kotak tangan tambahan. Pada {stats["two_candidates"]:,} gambar berhasil, percobaan yang dipakai mengembalikan lebih dari satu kandidat.',
        '- Hasil menurut jalur deteksi mencakup ' + ', '.join(f'`{key}` ({n:,})' for key, n in stats['passes'].most_common()) + '.',
        '- Metode pemilihan yang tercatat mencakup ' + ', '.join(f'`{key}` ({n:,})' for key, n in stats['methods'].most_common()) + '.',
        f'- MediaPipe mengklasifikasikan {stats["handedness"]["Left"]:,} tangan sebagai Left dan {stats["handedness"]["Right"]:,} sebagai Right. '
        f'Kesamaan dengan anotasi `leading_hand` {stats["lead_match"]:,}/{stats["lead_count"]:,} ({percent(stats["lead_match"], stats["lead_count"])}).',
        f'- {stats["out_of_frame"]:,} gambar berhasil memiliki setidaknya satu x/y di luar batas raster. Nilai dipertahankan untuk audit dan normalisasi.',
        f'- {visual_count} overlay menampilkan gambar asli, kotak anotasi, dan detail 21 landmark. [Sampel visual](landmark_visual_samples/)',
        '', '### Alasan kegagalan', '',
    ])
    for reason, count in stats['reasons'].most_common():
        breakdown = ', '.join(f'{label} {n:,}' for label, n in stats['per_class_reasons'][reason].most_common())
        lines.append(f'- `{reason}` — {count:,} ({breakdown}).')
    if not stats['reasons']:
        lines.append('- Total kegagalan ekstraksi 0.')
    if stats['errors']:
        lines.extend(['', '### Temuan validasi', ''])
        lines.extend(f'- `{kind}` — {count:,}' for kind, count in stats['errors'].most_common())
        lines.extend(f'- {detail}' for detail in stats['examples'])
    csv_size = sum((output / name).stat().st_size for name in ('landmarks_raw.csv', 'landmarks_normalized.csv', 'extraction_failures.csv'))
    lines.extend([
        '', '## Menjalankan ulang', '',
        f'Tiga CSV berukuran {csv_size / 2**20:.1f} MiB. Checkpoint SQLite di `ai/dataset/.landmark_cache/` menyimpan hasil inferensi per ID untuk melanjutkan proses dengan manifest, model, dan kode yang sama.',
        '', 'Jalankan dari akar proyek dengan Python 3.12.', '', '```sh',
        'python3.12 -m venv .venv', '.venv/bin/python -m pip install -r ai/requirements.txt',
        '.venv/bin/python ai/extract_landmarks.py', '.venv/bin/python ai/validate_landmarks.py', '```', '',
        'Sampel visual menggunakan gambar HaGRID 384p dan menambahkan overlay landmark. [Dataset sumber](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Lisensi HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)', '',
    ])
    with atomic_file(output / 'extraction_report.md') as file:
        file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true', help='Validasi tanpa menulis laporan dan overlay.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        parser.error('Folder output harus di luar folder gambar sumber.')
    try:
        spec = json.loads((output / 'feature_schema.json').read_text(encoding='utf-8'))
        stats, visual = validate(source, output, spec)
        if not args.check_only:
            visual_count = visual_samples(source, output, visual)
            report(output, spec, stats, visual_count)
            LOG.info('%d sampel visual dan laporan tersimpan di %s.', visual_count, output)
    except (csv.Error, KeyError, OSError, TypeError, ValueError) as error:
        LOG.error('Validasi berhenti: %s', error)
        return 1
    LOG.info('Berhasil per kelas: %s.', dict(stats['success']))
    LOG.info('Gagal per kelas: %s. Temuan validasi: %s.', dict(stats['failure']), dict(stats['errors']))
    return 1 if stats['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
