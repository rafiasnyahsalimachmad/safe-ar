#!/usr/bin/env python3
"""Validate the local export and create the contact sheet and Step 1 report."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import random
import zlib

from PIL import Image, ImageDraw, ImageFont, ImageOps
import pyarrow.parquet as pq

from hagrid_source import (
    DEFAULT_CACHE, DEFAULT_OUT, LOG, MAPPING, META_FIELDS, PREFIX, REVISION, SOURCE, VIEWER_SPLIT,
    annotation_errors, annotation_rows, atomic_file, configure_logging,
)

GESTURES = {
    'palm': 'Telapak menghadap kamera, lima jari terbuka',
    'fist': 'Tangan terkepal',
    'one': 'Satu telunjuk terangkat',
    'ok': 'Ibu jari dan telunjuk membentuk lingkaran',
    'three': 'Telunjuk, jari tengah, dan jari manis terangkat',
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}


def load_source(cache, output):
    for name in (REVISION, 'local'):
        folder = cache / name
        summary_path = folder / 'extraction.json'
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary['output'] == str(output.resolve()):
                parquet = folder / 'annotations_df.parquet'
                annotations = {}
                if parquet.is_file():
                    annotations = {
                        image_id: row
                        for image_id, row in annotation_rows(pq.ParquetFile(parquet))
                        if any(label in MAPPING for label in row['labels'] or ())
                    }
                return summary, annotations
    LOG.info('Snapshot sumber tidak ditemukan untuk folder ini. Memvalidasi ekspor secara lokal.')
    return {}, {}


def check_row(row):
    label = row['hagrid_label']
    safe_label = row['safe_ar_label']
    if label not in MAPPING or MAPPING[label] != safe_label:
        raise ValueError('Pemetaan label kelas tidak valid.')
    filename = f"{row['image_id']}.jpg"
    if PurePosixPath(filename).name != filename or '\\' in filename or not row['image_id']:
        raise ValueError('ID gambar tidak valid.')
    source_path = f'{PREFIX}/hagrid_120k/train_val_{label}/{filename}'
    if row['image_path'] != f'{safe_label}/{filename}' or row['source_image_path'] != source_path:
        raise ValueError('Path gambar tidak cocok dengan label atau ID.')
    if row['original_split'] != 'train_val' or row['hf_split'] != VIEWER_SPLIT:
        raise ValueError('Split tidak cocok dengan sumber.')
    annotation = json.loads(row['annotation_json'])
    if not isinstance(annotation, dict):
        raise ValueError('Format anotasi JSON tidak valid.')
    flat = {
        'bboxes': json.loads(row['bboxes']), 'labels': json.loads(row['labels']),
        'leading_hand': row['leading_hand'], 'leading_conf': float(row['leading_conf']), 'user_id': row['user_id'],
    }
    if flat != {key: annotation.get(key) for key in flat}:
        raise ValueError('Kolom anotasi tidak cocok dengan annotation_json.')
    issues = annotation_errors(annotation, label)
    if issues:
        raise ValueError(', '.join(issues))
    return annotation


def verify_image(path, row):
    data = path.read_bytes()
    if len(data) != int(row['source_size_bytes']) or zlib.crc32(data) != int(row['source_crc32'], 16):
        raise ValueError('Ukuran atau CRC-32 berkas berbeda dari sumber.')
    with Image.open(BytesIO(data)) as image:
        if image.format != 'JPEG':
            raise ValueError(f'Format gambar {image.format}, diharapkan JPEG.')
        image.load()
        size = image.size
        pixel_hash = sha256(image.convert('RGB').tobytes()).digest()
    return len(data), sha256(data).digest(), (size, pixel_hash), size


def duplicate_groups(groups):
    return [paths for paths in groups.values() if len(paths) > 1]


def validate(output, summary, source_annotations):
    metadata = output / 'metadata.csv'
    file_paths = {path.relative_to(output).as_posix() for path in output.rglob('*')
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path != output / 'contact_sheet.jpg'}
    paths, ids = set(), set()
    valid = defaultdict(list)
    byte_hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    errors = defaultdict(list)
    image_sizes, dimensions, box_labels = Counter(), Counter(), Counter()
    with metadata.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(META_FIELDS):
            raise ValueError('Header metadata.csv tidak cocok dengan skema ekspor.')
        for line, row in enumerate(reader, 2):
            relative = row['image_path']
            image_id = row['image_id']
            if relative in paths or image_id in ids:
                errors['path_or_id_duplicate'].append(f'Baris {line}: {relative}')
                continue
            paths.add(relative)
            ids.add(image_id)
            try:
                if any(value is None for value in row.values()) or None in row:
                    raise ValueError('Jumlah kolom CSV tidak sesuai.')
                annotation = check_row(row)
                if source_annotations and annotation != source_annotations.get(image_id):
                    raise ValueError('Anotasi berbeda dari Parquet sumber untuk ID ini.')
            except (ValueError, TypeError) as error:
                errors['metadata_invalid'].append(f'Baris {line}: {relative}: {error}')
                continue
            if relative not in file_paths:
                errors['missing_image'].append(relative)
                continue
            try:
                size, byte_hash, pixel_hash, dimensions_px = verify_image(output / relative, row)
            except (OSError, ValueError) as error:
                errors['corrupt_or_changed_image'].append(f'{relative}: {error}')
                continue
            safe_label = row['safe_ar_label']
            valid[safe_label].append(row)
            image_sizes[safe_label] += size
            byte_hashes[byte_hash].append(relative)
            pixel_hashes[pixel_hash].append(relative)
            dimensions[dimensions_px] += 1
            box_labels.update(annotation['labels'])
            if line % 5000 == 0:
                LOG.info('Metadata dan JPEG diperiksa sampai baris %s.', line)
    if extras := sorted(file_paths - paths):
        errors['unlisted_image'].extend(extras)
    for label in MAPPING.values():
        if not valid[label]:
            errors['empty_class'].append(label)
        expected = summary.get('extracted_counts', {}).get(label)
        if expected is not None and len(valid[label]) != expected:
            errors['count_mismatch'].append(f'{label}: {len(valid[label])} dari {expected}')
    duplicate_bytes = duplicate_groups(byte_hashes)
    duplicate_pixels = duplicate_groups(pixel_hashes)
    cross_class = [group for group in duplicate_pixels if len({path.split('/')[0] for path in group}) > 1]
    errors['cross_class_duplicate_pixels'].extend(' / '.join(paths) for paths in cross_class)
    return valid, image_sizes, dimensions, box_labels, errors, duplicate_bytes, duplicate_pixels


def crop_hand(image, row):
    labels, boxes = json.loads(row['labels']), json.loads(row['bboxes'])
    x, y, w, h = boxes[labels.index(row['hagrid_label'])]
    width, height = image.size
    margin = 0.16 * max(w * width, h * height)
    return image.crop((max(0, x * width - margin), max(0, y * height - margin),
                       min(width, (x + w) * width + margin), min(height, (y + h) * height + margin)))


def paste_preview(sheet, image, xy, size):
    image = ImageOps.contain(image, size)
    sheet.paste(image, (xy[0] + (size[0] - image.width) // 2, xy[1] + (size[1] - image.height) // 2))


def contact_sheet(output, valid):
    rng = random.Random(42)
    width, height, columns = 305, 234, 4
    sheet = Image.new('RGB', (columns * width + 24, len(MAPPING) * height + 36), '#101828')
    draw = ImageDraw.Draw(sheet)
    heading, caption = ImageFont.load_default(18), ImageFont.load_default(13)
    draw.text((12, 9), 'S.A.F.E. AR  |  HaGRID  |  Foto sumber dan detail tangan beranotasi', font=heading, fill='white')
    samples = {}
    for row_number, (label, safe_label) in enumerate(MAPPING.items()):
        rows = valid[safe_label]
        samples[safe_label] = rng.sample(rows, min(columns, len(rows)))
        top = 38 + row_number * height
        draw.text((12, top), f'{safe_label} / {label}', font=heading, fill='#b8dbff')
        for column, row in enumerate(samples[safe_label]):
            left = 12 + column * width
            draw.rounded_rectangle((left, top + 23, left + width - 9, top + height - 7), radius=6, fill='#253149')
            with Image.open(output / row['image_path']) as original:
                image = original.convert('RGB')
                paste_preview(sheet, image, (left + 4, top + 28), (175, 169))
                paste_preview(sheet, crop_hand(image, row), (left + 181, top + 28), (108, 169))
            draw.text((left + 8, top + 201), row['image_id'], font=caption, fill='white')
    with atomic_file(output / 'contact_sheet.jpg', 'wb') as file:
        sheet.save(file, format='JPEG', quality=92)
    return samples


def report(output, summary, source_checked, result, samples):
    valid, sizes, dimensions, box_labels, errors, byte_duplicates, pixel_duplicates = result
    images = sum(map(len, valid.values()))
    status = 'Lulus' if not any(errors.values()) else 'Gagal'
    image_bytes = sum(sizes.values())
    csv_bytes = (output / 'metadata.csv').stat().st_size
    sheet_bytes = (output / 'contact_sheet.jpg').stat().st_size
    lines = [
        '# S.A.F.E. AR — Step 1. Inspeksi dan ekstraksi HaGRID', '',
        f'**Validasi otomatis {status.lower()}**. {images:,} JPEG tervalidasi di `ai/dataset/hagrid_filtered/`.'
        if output == DEFAULT_OUT else f'**Validasi otomatis {status.lower()}**. {images:,} JPEG tervalidasi di `{output.resolve()}`.', '',
        '## Pemetaan gestur dan jumlah JPEG', '',
        '| Kelas S.A.F.E. AR | HaGRID | Kecocokan visual | Folder sumber | Lolos validasi | Ukuran JPEG (byte) |',
        '|---|---|---|---:|---:|---:|',
    ]
    for label, safe_label in MAPPING.items():
        available = summary.get('image_counts', {}).get(label)
        source_count = f'{available:,}' if available is not None else '—'
        lines.append(f'| {safe_label} | `{label}` | {GESTURES[label]} | {source_count} | {len(valid[safe_label]):,} | {sizes[safe_label]:,} |')
    lines.extend([
        f'| **Total** | | | | **{images:,}** | **{image_bytes:,}** |', '',
        'Contoh sumber memperlihatkan `palm` dengan jari terbuka dan `stop` dengan jari rapat. Pemetaan DANGER_STOP memakai `palm`.',
        'Contact sheet menampilkan sampai empat foto dan detail kotak tangan target per kelas. UUID di bawah foto mengacu pada `image_id` di `metadata.csv`.',
        '', '## Sumber, split, dan anotasi', '',
    ])
    if summary:
        labels = ', '.join(f'`{label}`' for label in summary['annotation_labels'])
        lines.extend([
            f'- Revisi Hugging Face `{summary["revision"]}`.' if summary['revision'] else '- ZIP sumber lokal.',
            f'- Seluruh label yang ditemukan pada anotasi ({len(summary["annotation_labels"])}): {labels}.',
            f'- ZIP berisi {summary["image_total"]:,} JPEG dalam {len(summary["image_counts"])} folder kelas. '
            f'`no_gesture` muncul pada {summary["annotation_labels"].get("no_gesture", 0):,} kotak tangan dan tidak memiliki folder gambar tersendiri.',
            f'- ZIP menyediakan {summary["annotation_rows"]:,} baris dalam {summary["annotation_row_groups"]} row group Parquet serta {len(summary["json_members"])} berkas JSON kelas di `ann_train_val/`. '
            'Ekstraksi memakai Parquet dari dalam ZIP dan mencocokkan UUID gambar dengan indeks anotasi.',
        ])
    else:
        lines.append('- Rincian inventaris sumber tersedia setelah menjalankan `inspect_hagrid.py` dan `filter_hagrid.py` pada dataset ini.')
    lines.extend([
        '- `original_split=train_val` mengikuti nama folder ZIP. `hf_split=train` merekam nama split di viewer Hugging Face. Sampel ini menggabungkan train dan val tanpa tanda per gambar serta tidak menyediakan split test.',
        '- `bboxes` memuat `[x, y, lebar, tinggi]` dengan titik asal kiri atas dan koordinat ternormalisasi. `labels` berurutan sesuai kotak. CSV juga menyimpan `leading_hand`, `leading_conf`, `user_id`, anotasi lengkap, path sumber, ukuran, dan CRC-32 setiap JPEG.',
        f'- Hasil memuat {sum(box_labels.values()):,} kotak tangan, termasuk {box_labels["no_gesture"]:,} berlabel `no_gesture`.',
        '- Dataset card dan API viewer hanya menyediakan pratinjau. Endpoint Parquet publik tidak mengembalikan berkas dan filter viewer dinonaktifkan. Ekstraksi jaringan memakai HTTP Range pada ZIP dan menyimpan anggota JPEG terpilih tanpa pengodean ulang.',
        f'- [Dataset card]({SOURCE}) · [Inventaris ZIP]({SOURCE}/tree/{REVISION}) · [HaGRID v1](https://raw.githubusercontent.com/hukenovs/hagrid/Hagrid_v1/README.md)',
        '', '## Validasi dan kendala', '',
        f'- {images:,} JPEG lolos dekode, CRC-32, dan pemeriksaan ukuran. Dimensi asli yang ditemukan mencakup {len(dimensions)} ukuran; yang paling sering adalah '
        + ', '.join(f'{w}×{h} ({n:,})' for (w, h), n in dimensions.most_common(4)) + '.',
        f'- Duplikat persis: {len(byte_duplicates):,} grup byte identik dan {len(pixel_duplicates):,} grup piksel RGB identik. '
        'Pencarian duplikat memakai SHA-256 dan mencakup seluruh kelas.',
        '- Path, ID, label, split, jumlah kotak, rentang koordinat, dan kolom anotasi diperiksa pada setiap baris.',
        f'- Anotasi juga dicocokkan dengan Parquet sumber berdasarkan UUID ({source_checked:,} gambar lolos).' if source_checked else '- Validasi anotasi memakai struktur dan konsistensi CSV lokal.',
        f'- {sum(len(rows) for rows in samples.values())} contoh tersimpan di [contact_sheet.jpg](contact_sheet.jpg).',
    ])
    if summary.get('source_issues'):
        issues = summary['source_issues']
        lines.append(f'- Di seluruh ZIP, {issues.get("target_label_missing", 0)} anotasi tidak memuat label folder dan {issues.get("bbox_invalid", 0)} kotak berada di luar batas yang ditoleransi.')
    if excluded := summary.get('excluded'):
        lines.append(f'- {len(excluded)} gambar dari folder target dikecualikan karena anotasinya tidak valid untuk kelas tersebut.')
        for item in excluded:
            lines.append(f'  - `{item["source_image_path"]}` ({", ".join(item["issues"])})')
    for kind, entries in sorted(errors.items()):
        if entries:
            lines.append(f'- `{kind}`: {len(entries):,} temuan.')
            lines.extend(f'  - {entry}' for entry in entries[:10])
    for title, groups in [('Byte identik', byte_duplicates), ('Piksel identik', pixel_duplicates)]:
        for group in groups[:10]:
            lines.append(f'- {title}: ' + ', '.join(f'`{path}`' for path in group))
    lines.extend([
        '', '## Ukuran dan lokasi', '',
        f'- JPEG: {image_bytes:,} byte ({image_bytes / 2**20:,.1f} MiB).',
        f'- [metadata.csv](metadata.csv): {csv_bytes:,} byte. [Contact sheet](contact_sheet.jpg): {sheet_bytes:,} byte.',
        f'- Total JPEG, CSV, dan contact sheet: **{image_bytes + csv_bytes + sheet_bytes:,} byte ({(image_bytes + csv_bytes + sheet_bytes) / 2**20:,.1f} MiB)**.',
        '', '## Lisensi dan atribusi', '',
        '- Dataset card mencantumkan CC BY-SA 4.0. Saat membagikan materi, sertakan atribusi dan tautan lisensi, nyatakan perubahan, serta gunakan lisensi yang sama atau kompatibel untuk adaptasi yang dibagikan. Lisensi standar juga melarang pembatasan tambahan atas hak yang diberikannya. [Dataset card](' + SOURCE + ') · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)',
        '- Repositori asli HaGRID merujuk ke lisensi publik tersendiri yang diadaptasi dari CC BY-SA 4.0. Dokumen tersebut mengizinkan reproduksi dan adaptasi, mewajibkan atribusi dan pemberitahuan perubahan saat dibagikan, serta mensyaratkan lisensi dengan unsur setara atau kompatibel untuk adaptasi yang dibagikan. Hak privasi dan hak atas penggambaran orang berada di luar cakupan pemberian lisensinya. [Lisensi spesifik HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)',
        '- Kredit HaGRID kepada Alexander Kapitanov, Andrey Makhlyarchuk, dan Karina Kvanchiani. cj-mills membuat sampel ukuran 384p. S.A.F.E. AR memilih lima kelas, memetakan label, mengekspor anotasi, dan membuat contact sheet. [Dataset card](' + SOURCE + ') · [Publikasi HaGRID](https://arxiv.org/abs/2206.08219)',
        '', '## Menjalankan ulang', '',
        'Jalankan dari akar repository dengan lingkungan Python 3.12.', '',
        '```sh',
        'python3.12 -m venv .venv',
        '.venv/bin/python -m pip install -r ai/requirements.txt',
        '.venv/bin/python ai/inspect_hagrid.py',
        '.venv/bin/python ai/filter_hagrid.py',
        '.venv/bin/python ai/validate_dataset.py',
        '```', '',
        'Ekstraksi melanjutkan unduhan dengan melewati JPEG yang ukuran dan CRC-32-nya sudah cocok. Gunakan `--archive /path/ke/sumber.zip` pada inspeksi dan ekstraksi untuk membaca ZIP yang sudah tersedia secara lokal. Validasi berjalan tanpa jaringan.',
        '',
    ])
    with atomic_file(output / 'dataset_report.md', 'w') as file:
        file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT, help='Folder ekspor yang akan diperiksa.')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE, help='Cache anotasi sumber untuk pengecekan tambahan.')
    parser.add_argument('--check-only', action='store_true', help='Validasi tanpa menulis laporan dan contact sheet.')
    args = parser.parse_args()
    configure_logging()
    try:
        summary, source_annotations = load_source(args.cache, args.output)
        result = validate(args.output, summary, source_annotations)
        valid, _, _, _, errors, byte_duplicates, pixel_duplicates = result
        if not args.check_only:
            samples = contact_sheet(args.output, valid)
            source_checked = sum(map(len, valid.values())) if source_annotations else 0
            report(args.output, summary, source_checked, result, samples)
            LOG.info('Laporan dan contact sheet tersimpan di %s.', args.output)
        for kind, entries in errors.items():
            if entries:
                LOG.error('%s: %s, contoh: %s', kind, len(entries), entries[0])
        LOG.info('JPEG tervalidasi per kelas: %s.', {key: len(valid[key]) for key in MAPPING.values()})
        LOG.info('Grup duplikat byte: %s. Grup duplikat piksel: %s.', len(byte_duplicates), len(pixel_duplicates))
        return 1 if any(errors.values()) else 0
    except (OSError, ValueError) as error:
        LOG.error('Validasi gagal: %s', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
