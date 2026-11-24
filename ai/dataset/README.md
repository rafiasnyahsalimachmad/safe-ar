# Dataset lokal

Letakkan salinan lengkap dataset dan hasil eksperimen di bawah folder ini. Git mengabaikan subfolder data dan cache berikut.

| Folder | Isi |
| --- | --- |
| `hagrid_filtered/` | Lima folder kelas, `metadata.csv`, contact sheet, dan laporan Step 1 |
| `processed/` | `landmarks_raw.csv`, `landmarks_normalized.csv`, `extraction_failures.csv`, skema, laporan, dan overlay Step 2 |
| `.hagrid_cache/` | Indeks ZIP dan anotasi Parquet HaGRID, dibuat saat inspeksi atau filtering |
| `.landmark_cache/` | Checkpoint SQLite ekstraksi landmark, dibuat oleh Step 2 |

`image_path` pada ketiga CSV Step 2 relatif terhadap `hagrid_filtered/`. Simpan `metadata.csv` bersama folder gambar agar label, bounding box, split, dan ID subjek tetap dapat diverifikasi. Gunakan `--output` dengan folder baru untuk menghasilkan ekspor lain sambil mempertahankan hasil ini.

[Instruksi pipeline dan lisensi](../../README.md#dataset-lisensi-dan-hasil)
