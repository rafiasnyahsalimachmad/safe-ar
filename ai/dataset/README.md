# Dataset S.A.F.E. AR

| Path | Isi yang tersedia setelah checkout |
| --- | --- |
| [`hagrid_filtered/`](hagrid_filtered/) | 35.134 JPEG HaGRID dalam lima folder kelas, beserta contact sheet dan laporan |
| [`hagrid_filtered/metadata.csv`](hagrid_filtered/metadata.csv) | Metadata dan anotasi 35.134 gambar HaGRID terfilter |
| [`processed/landmarks_raw.csv`](processed/landmarks_raw.csv) | Landmark mentah dari 34.061 gambar |
| [`processed/landmarks_normalized.csv`](processed/landmarks_normalized.csv) | Vektor ternormalisasi dari 34.061 gambar |
| [`processed/extraction_failures.csv`](processed/extraction_failures.csv) | 1.073 rekaman kegagalan ekstraksi |
| [`processed/feature_schema.json`](processed/feature_schema.json) | Kontrak fitur yang dibaca validator landmark |

`image_path` pada CSV relatif terhadap `hagrid_filtered/`. [Overlay landmark](processed/landmark_visual_samples/) tersedia bersama laporan ekstraksi. JPEG dan overlay memakai sekitar 859 MiB sebelum kompresi Git. Cache indeks ZIP dan checkpoint dibuat secara lokal oleh masing-masing tahap.

JPEG 384p dipertahankan sesuai byte sampel sumber. CSV dan pratinjau merupakan hasil adaptasi S.A.F.E. AR dari HaGRID oleh Alexander Kapitanov, Andrey Makhlyarchuk, dan Karina Kvanchiani, menggunakan sampel 384p oleh cj-mills. Proyek ini memilih lima kelas, memetakan label, mengekspor anotasi, dan menghasilkan fitur MediaPipe. [Dataset sumber](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Lisensi HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

[Laporan Step 1](../../docs/dataset_report.md) · [Laporan Step 2](../../docs/extraction_report.md) · [Skema fitur](../../docs/feature_schema.json)
