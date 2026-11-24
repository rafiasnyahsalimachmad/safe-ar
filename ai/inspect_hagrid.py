#!/usr/bin/env python3
"""Inspect the published ZIP and its annotations without downloading image sets."""

import argparse
import json
from urllib.request import urlopen
import zipfile

from hagrid_source import Archive, LOG, MAPPING, REPO, REVISION, SOURCE, configure_logging, read_inventory, source_args, write_json


def viewer_info(archive):
    urls = {
        'dataset_card': f'{SOURCE}/raw/{REVISION}/README.md',
        'splits': f'https://datasets-server.huggingface.co/splits?dataset={REPO}',
        'parquet': f'https://datasets-server.huggingface.co/parquet?dataset={REPO}',
        'viewer': f'https://datasets-server.huggingface.co/is-valid?dataset={REPO}',
    }
    result = {}
    for key, url in urls.items():
        try:
            with urlopen(url, context=archive.ssl_context, timeout=20) as response:
                result[key] = response.read().decode('utf-8') if key == 'dataset_card' else json.load(response)
        except (OSError, ValueError) as error:
            LOG.warning('Metadata Hugging Face %s tidak dapat dibaca: %s', key, error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_args(parser)
    parser.add_argument('--skip-viewer', action='store_true', help='Lewati endpoint dataset card dan viewer.')
    args = parser.parse_args()
    configure_logging()
    try:
        archive = Archive(args.cache, args.archive)
        _, _, summary = read_inventory(archive)
        if not args.skip_viewer and args.archive is None:
            summary['hugging_face'] = viewer_info(archive)
        summary.update(range_requests=archive.requests, range_bytes=archive.downloaded)
        write_json(archive.cache / 'inspection.json', summary)
        LOG.info('Gambar per folder: %s', summary['image_counts'])
        LOG.info('Label per kotak: %s', summary['annotation_labels'])
        LOG.info('Split ZIP: %s. Split viewer HF: %s.', summary['source_splits'], summary['hf_splits'])
        LOG.info('Skema Parquet: %s', summary['annotation_columns'])
        LOG.info('Pemetaan S.A.F.E. AR: %s. Gambar layak: %s.', MAPPING, summary['extractable_counts'])
        LOG.info('Masalah sumber: %s. Gambar target dikecualikan: %s.', summary['source_issues'], len(summary['excluded']))
        LOG.info('Inspeksi tersimpan di %s. Unduhan rentang: %s byte.', archive.cache / 'inspection.json', archive.downloaded)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        LOG.error('Inspeksi gagal: %s', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
