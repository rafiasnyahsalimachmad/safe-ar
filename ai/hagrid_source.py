"""Shared source access and annotation checks for the HaGRID Step 1 scripts."""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException
import io
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import ssl
import struct
from tempfile import NamedTemporaryFile
from threading import Lock
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

import pyarrow.parquet as pq
import truststore

DATASET = Path(__file__).resolve().parent / 'dataset'
DEFAULT_OUT = DATASET / 'hagrid_filtered'
DEFAULT_CACHE = DATASET / '.hagrid_cache'
REPO = 'cj-mills/hagrid-sample-120k-384p'
SOURCE = f'https://huggingface.co/datasets/{REPO}'
REVISION = '61d5663f48cf742257c0ec3666a5bda8a19bc86f'
PREFIX = 'hagrid-sample-120k-384p'
ARCHIVE_URL = f'{SOURCE}/resolve/{REVISION}/{PREFIX}.zip?download=true'
ARCHIVE_SIZE = 3_487_867_278
ZIP_INDEX_LIMIT = 64 * 1024 * 1024
ANNOTATION_MEMBER = f'{PREFIX}/annotations_df.parquet'
ANNOTATION_ID = '__index_level_0__'
VIEWER_SPLIT = 'train'
MAPPING = {
    'palm': 'DANGER_STOP',
    'fist': 'NEED_MEDICAL',
    'one': 'EVACUATION',
    'ok': 'SAFE',
    'three': 'NEED_LOGISTICS',
}
META_FIELDS = (
    'image_path', 'hagrid_label', 'safe_ar_label', 'original_split', 'hf_split',
    'image_id', 'source_image_path', 'source_size_bytes', 'source_crc32',
    'bboxes', 'labels', 'leading_hand', 'leading_conf', 'user_id', 'annotation_json',
)
ANNOTATION_FIELDS = ('bboxes', 'labels', 'leading_hand', 'leading_conf', 'user_id')
LOG = logging.getLogger('hagrid')


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def source_args(parser: argparse.ArgumentParser):
    parser.add_argument('--archive', type=Path, help='Gunakan ZIP sumber yang sudah tersedia secara lokal.')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE, help='Folder cache indeks ZIP dan anotasi.')


@contextmanager
def atomic_file(path, mode):
    options = {'encoding': 'utf-8', 'newline': ''} if mode == 'w' else {}
    with NamedTemporaryFile(mode, dir=path.parent, delete=False, **options) as temporary:
        temp_path = Path(temporary.name)
        try:
            yield temporary
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def write_bytes(path, data):
    with atomic_file(path, 'wb') as file:
        file.write(data)


def write_json(path, data):
    with atomic_file(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write('\n')


class ZipDirectory(io.BytesIO):
    """Expose the cached ZIP directory at its original position to zipfile."""

    def __init__(self, data, archive_size):
        super().__init__(data)
        self.start = archive_size - len(data)

    def tell(self):
        return self.start + super().tell()

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            offset -= self.start
        return self.start + super().seek(offset, whence)


class Archive:
    def __init__(self, cache, local_path=None):
        self.local_path = local_path
        self.cache = cache / (REVISION if local_path is None else 'local')
        self.cache.mkdir(parents=True, exist_ok=True)
        self.size = local_path.stat().st_size if local_path else ARCHIVE_SIZE
        self.downloaded = 0
        self.requests = 0
        self._lock = Lock()
        self.ssl_context = None if local_path else truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if local_path:
            with zipfile.ZipFile(local_path) as archive:
                self.members = archive.infolist()
                self._directory_start = archive.start_dir
        else:
            self.members = self._remote_directory()
        self.members.sort(key=lambda item: item.header_offset)
        if not self.members:
            raise ValueError('ZIP tidak berisi anggota.')
        self.by_name = {member.filename: member for member in self.members}
        if len(self.by_name) != len(self.members):
            raise ValueError('ZIP memiliki nama anggota duplikat.')
        self.ends = {
            current.filename: following.header_offset
            for current, following in zip(self.members, self.members[1:])
        }
        last = self.members[-1]
        self.ends[last.filename] = self._directory_start

    def _remote_directory(self):
        cached_path = self.cache / 'index.zip-tail'
        if not cached_path.exists():
            tail = self.read_block(self.size - 65_557, self.size)
            signature = tail.rfind(b'PK\x05\x06')
            if signature < 0 or signature + 22 > len(tail):
                raise ValueError('ZIP end-of-central-directory tidak ditemukan.')
            start = struct.unpack_from('<I', tail, signature + 16)[0]
            if not self.size - ZIP_INDEX_LIMIT <= start < self.size - 22:
                raise ValueError('Offset direktori ZIP tidak valid untuk revisi ini.')
            data = self.read_block(start, self.size)
            members = self._parse_directory(data)
            write_bytes(cached_path, data)
            return members
        try:
            return self._parse_directory(cached_path.read_bytes())
        except (ValueError, zipfile.BadZipFile):
            LOG.warning('Cache indeks ZIP rusak, mengambil indeks kembali.')
            cached_path.unlink()
            return self._remote_directory()

    def _parse_directory(self, data):
        with zipfile.ZipFile(ZipDirectory(data, self.size)) as archive:
            self._directory_start = archive.start_dir
            return archive.infolist()

    def read_block(self, start, end):
        """Read the half-open byte range [start, end). Reject whole-file replies."""
        if not 0 <= start < end <= self.size:
            raise ValueError(f'Rentang ZIP tidak valid [{start}, {end}).')
        size = end - start
        if self.local_path:
            with self.local_path.open('rb') as file:
                file.seek(start)
                data = file.read(size)
            if len(data) != size:
                raise OSError(f'Rentang ZIP lokal terpotong [{start}, {end}).')
            return data
        byte_range = f'{start}-{end - 1}'
        expected = f'bytes {byte_range}/{self.size}'
        for attempt in range(4):
            url = ARCHIVE_URL if attempt == 0 else f'{ARCHIVE_URL}&byte_range={byte_range}&retry={attempt}'
            request = Request(url, headers={'Range': f'bytes={byte_range}', 'Accept-Encoding': 'identity'})
            try:
                with urlopen(request, timeout=45, context=self.ssl_context) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise OSError(f'HTTP {response.status}, Content-Range tidak cocok dengan {expected}.')
                    data = response.read(size + 1)
                with self._lock:
                    self.downloaded += len(data)
                    self.requests += 1
                if len(data) != size:
                    raise OSError(f'Respons hanya berisi {len(data)} dari {size} byte.')
                return data
            except (OSError, HTTPException) as error:
                if attempt == 3:
                    raise OSError(f'Gagal mengakses rentang ZIP {byte_range}: {error}') from error
                LOG.warning('Rentang %s gagal (%s), mencoba kembali.', byte_range, error)
                time.sleep(2 ** attempt)

    def read_member(self, member):
        start = member.header_offset
        block = self.read_block(start, self.ends[member.filename])
        return member_bytes(member, block, start)

    def annotation_file(self):
        member = self.by_name.get(ANNOTATION_MEMBER)
        if member is None:
            raise ValueError(f'Anotasi sumber tidak tersedia: {ANNOTATION_MEMBER}')
        cache_path = self.cache / 'annotations_df.parquet'
        if not matches_zip(cache_path, member):
            LOG.info('Membaca Parquet anotasi dari ZIP (%s byte).', member.file_size)
            write_bytes(cache_path, self.read_member(member))
        return pq.ParquetFile(cache_path)


def member_bytes(member, block, start):
    offset = member.header_offset - start
    signature, _, flags, compression, _, _, _, _, _, name_size, extra_size = struct.unpack_from(
        '<4s5H3I2H', block, offset,
    )
    name_start = offset + 30
    name_end = name_start + name_size
    name = block[name_start:name_end].decode('utf-8' if flags & 0x800 else 'cp437')
    if signature != b'PK\x03\x04' or flags & 1 or compression != member.compress_type or name != member.filename:
        raise ValueError(f'Header ZIP tidak cocok: {member.filename}')
    data_start = name_end + extra_size
    data_end = data_start + member.compress_size
    if data_end > len(block):
        raise ValueError(f'Anggota ZIP terpotong: {member.filename}')
    data = block[data_start:data_end]
    if compression == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif compression != zipfile.ZIP_STORED:
        raise ValueError(f'Kompresi ZIP {compression} tidak didukung: {name}')
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError(f'Ukuran atau CRC-32 tidak cocok: {name}')
    return data


def matches_zip(path, member):
    try:
        return path.stat().st_size == member.file_size and zlib.crc32(path.read_bytes()) == member.CRC
    except FileNotFoundError:
        return False


@dataclass(frozen=True)
class ImageEntry:
    member: zipfile.ZipInfo
    label: str
    split: str
    image_id: str
    filename: str

    @property
    def output_path(self):
        return f'{MAPPING[self.label]}/{self.filename}'


def image_entries(archive):
    entries = {}
    for member in archive.members:
        path = PurePosixPath(member.filename)
        if len(path.parts) != 4 or path.parts[:2] != (PREFIX, 'hagrid_120k') or path.suffix.lower() != '.jpg':
            continue
        folder = path.parts[2]
        if not folder.startswith('train_val_'):
            raise ValueError(f'Folder gambar sumber tidak dikenal: {folder}')
        if path.stem in entries:
            raise ValueError(f'ID gambar sumber duplikat: {path.stem}')
        entries[path.stem] = ImageEntry(member, folder.removeprefix('train_val_'), 'train_val', path.stem, path.name)
    if not entries:
        raise ValueError('ZIP tidak memiliki JPEG HaGRID pada struktur yang diharapkan.')
    return entries


def annotation_rows(parquet):
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            image_id = row.pop(ANNOTATION_ID)
            yield image_id, row


def annotation_errors(annotation, label):
    labels, boxes = annotation.get('labels'), annotation.get('bboxes')
    issues = []
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
        issues.append('labels_invalid')
    else:
        if label not in labels:
            issues.append('target_label_missing')
        if set(labels) - {label, 'no_gesture'}:
            issues.append('other_gesture_label')
    if not isinstance(boxes, list) or not boxes or not isinstance(labels, list) or len(boxes) != len(labels):
        issues.append('bbox_label_count_invalid')
    elif any(not valid_box(box) for box in boxes):
        issues.append('bbox_invalid')
    if annotation.get('leading_hand') not in {'left', 'right'}:
        issues.append('leading_hand_invalid')
    confidence = annotation.get('leading_conf')
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        issues.append('leading_conf_invalid')
    if not isinstance(annotation.get('user_id'), str) or not annotation['user_id']:
        issues.append('user_id_invalid')
    return issues


def valid_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        return False
    x, y, width, height = box
    return x >= 0 and y >= 0 and width > 0 and height > 0 and x + width <= 1.000001 and y + height <= 1.000001


def read_inventory(archive):
    entries = image_entries(archive)
    parquet = archive.annotation_file()
    image_counts = Counter(entry.label for entry in entries.values())
    box_labels = Counter()
    source_issues = Counter()
    bad_selected = []
    selected = {}
    selected_counts = Counter()
    seen = set()
    for image_id, annotation in annotation_rows(parquet):
        if image_id in seen:
            raise ValueError(f'ID anotasi duplikat: {image_id}')
        seen.add(image_id)
        box_labels.update(annotation['labels'] or [])
        entry = entries.get(image_id)
        if entry is None:
            source_issues['annotation_without_image'] += 1
            continue
        issues = annotation_errors(annotation, entry.label)
        source_issues.update(issues)
        if entry.label in MAPPING:
            if issues:
                bad_selected.append({'source_image_path': entry.member.filename, 'issues': issues})
            else:
                selected[image_id] = annotation
                selected_counts[entry.label] += 1
    for image_id in entries.keys() - seen:
        entry = entries[image_id]
        source_issues['image_without_annotation'] += 1
        if entry.label in MAPPING:
            bad_selected.append({'source_image_path': entry.member.filename, 'issues': ['image_without_annotation']})
    summary = {
        'source': SOURCE, 'revision': REVISION if archive.local_path is None else None,
        'archive_bytes': archive.size,
        'image_counts': dict(sorted(image_counts.items())),
        'image_total': len(entries),
        'annotation_labels': dict(sorted(box_labels.items())),
        'annotation_rows': parquet.metadata.num_rows,
        'annotation_row_groups': parquet.num_row_groups,
        'annotation_columns': {field.name: str(field.type) for field in parquet.schema_arrow},
        'json_members': sorted(name for name in archive.by_name if '/ann_train_val/' in name and name.endswith('.json')),
        'source_splits': sorted({entry.split for entry in entries.values()}),
        'hf_splits': [VIEWER_SPLIT],
        'source_issues': dict(source_issues),
        'excluded': sorted(bad_selected, key=lambda row: row['source_image_path']),
        'extractable_counts': {label: selected_counts[label] for label in MAPPING},
    }
    return entries, selected, summary
