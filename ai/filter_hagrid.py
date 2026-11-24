#!/usr/bin/env python3
"""Extract the five S.A.F.E. AR gestures and preserve source bytes and annotations."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
from pathlib import Path
import struct
import zipfile
import zlib

from hagrid_source import (
    ANNOTATION_FIELDS, Archive, DEFAULT_OUT, LOG, MAPPING, META_FIELDS, VIEWER_SPLIT,
    atomic_file, configure_logging, matches_zip, member_bytes, read_inventory, source_args,
    write_bytes, write_json,
)

BLOCK_SIZE = 8 * 1024 * 1024


def blocks(archive, pending):
    """Batch adjacent missing JPEG entries. Gaps can contain other or saved images."""
    start = end = 0
    group = []
    for entry in pending:
        member = entry.member
        member_end = archive.ends[member.filename]
        if group and (member.header_offset != end or member_end - start > BLOCK_SIZE):
            yield start, end, group
            group = []
        if not group:
            start = member.header_offset
        end = member_end
        group.append(entry)
    if group:
        yield start, end, group


def save_block(archive, output, block_info):
    start, end, entries = block_info
    data = archive.read_block(start, end)
    for entry in entries:
        payload = member_bytes(entry.member, data, start)
        write_bytes(output / entry.output_path, payload)
    return len(entries)


def fetch_pending(archive, output, pending, workers):
    jobs = iter(blocks(archive, pending))
    saved = 0
    checkpoint = 2_000
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {pool.submit(save_block, archive, output, job) for job in take(jobs, workers)}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for job in done:
                saved += job.result()
                if saved >= checkpoint or saved == len(pending):
                    LOG.info('Gambar baru tersimpan: %s/%s. Unduhan: %.1f MiB.', saved, len(pending), archive.downloaded / 2**20)
                    checkpoint += 2_000
                next_block = next(jobs, None)
                if next_block is not None:
                    active.add(pool.submit(save_block, archive, output, next_block))
    return saved


def take(iterator, count):
    for _ in range(count):
        value = next(iterator, None)
        if value is None:
            return
        yield value


def metadata_row(entry, annotation):
    member = entry.member
    row = {
        'image_path': entry.output_path, 'hagrid_label': entry.label, 'safe_ar_label': MAPPING[entry.label],
        'original_split': entry.split, 'hf_split': VIEWER_SPLIT, 'image_id': entry.image_id,
        'source_image_path': member.filename, 'source_size_bytes': member.file_size,
        'source_crc32': f'{member.CRC:08x}',
        'annotation_json': json.dumps(annotation, ensure_ascii=False, separators=(',', ':')),
    }
    for field in ANNOTATION_FIELDS:
        value = annotation[field]
        row[field] = json.dumps(value, separators=(',', ':')) if isinstance(value, list) else value
    return row


def export(archive, output, workers):
    entries, annotations, summary = read_inventory(archive)
    missing = set(MAPPING) - set(summary['image_counts'])
    if missing:
        LOG.warning('Label target tidak tersedia: %s', ', '.join(sorted(missing)))
    output.mkdir(parents=True, exist_ok=True)
    for safe_label in MAPPING.values():
        (output / safe_label).mkdir(exist_ok=True)
    selected = sorted((entries[key] for key in annotations), key=lambda entry: entry.member.header_offset)
    pending = [entry for entry in selected if not matches_zip(output / entry.output_path, entry.member)]
    LOG.info('%s gambar layak, %s sudah cocok CRC sumber, %s perlu dibaca.', len(selected), len(selected) - len(pending), len(pending))
    for row in summary['excluded']:
        LOG.warning('Melewati %s: %s', row['source_image_path'], ', '.join(row['issues']))
    saved = fetch_pending(archive, output, pending, workers)
    with atomic_file(output / 'metadata.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS)
        writer.writeheader()
        for entry in sorted(selected, key=lambda entry: entry.output_path):
            writer.writerow(metadata_row(entry, annotations[entry.image_id]))
    counts = Counter(MAPPING[entry.label] for entry in selected)
    summary.update(
        extracted_counts={label: counts[label] for label in MAPPING.values()},
        target_image_bytes=sum(entry.member.file_size for entry in selected),
        range_requests=archive.requests, range_bytes=archive.downloaded,
        output=str(output.resolve()), saved_this_run=saved,
    )
    write_json(archive.cache / 'extraction.json', summary)
    LOG.info('Ekstraksi selesai: %s. Unduhan HTTP Range: %s byte dalam %s permintaan.', dict(counts), archive.downloaded, archive.requests)
    return 1 if missing or any(not counts[label] for label in MAPPING.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_args(parser)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT, help='Folder hasil ekstraksi.')
    parser.add_argument('--workers', type=int, default=8, help='Jumlah unduhan rentang paralel.')
    parser.add_argument('--repair', action='store_true', help='Periksa ulang ekspor yang ada dan pulihkan berkas dari sumber bila perlu.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers harus antara 1 dan 32.')
    configure_logging()
    if (args.output / 'metadata.csv').exists() and not args.repair:
        LOG.info('Ekspor sudah ada di %s. Gunakan validator, atau pilih --output yang baru.', args.output.resolve())
        return 0
    try:
        return export(Archive(args.cache, args.archive), args.output, args.workers)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, zlib.error, struct.error) as error:
        LOG.error('Ekstraksi gagal: %s. Gambar tersimpan dapat dipakai saat script dijalankan ulang.', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
