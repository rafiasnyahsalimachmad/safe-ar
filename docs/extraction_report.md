# S.A.F.E. AR Kelompok 6 — Step 2

Tautan ke CSV atau gambar mengarah ke salinan lokal yang disimpan di luar Git.

**Validasi dataset fitur lulus.** 34,061 dari 35,134 gambar berhasil (96.95%). 1,073 gagal diekspor sebagai fitur dan dicatat di [extraction_failures.csv](../ai/dataset/processed/extraction_failures.csv).

## Jumlah dan distribusi kelas

| S.A.F.E. AR | HaGRID | Diproses | Berhasil | Gagal | Tingkat berhasil | Porsi fitur akhir |
|---|---|---:|---:|---:|---:|---:|
| DANGER_STOP | `palm` | 7,082 | 7,023 | 59 | 99.17% | 20.62% |
| NEED_MEDICAL | `fist` | 6,940 | 6,551 | 389 | 94.39% | 19.23% |
| EVACUATION | `one` | 7,111 | 6,861 | 250 | 96.48% | 20.14% |
| SAFE | `ok` | 6,999 | 6,869 | 130 | 98.14% | 20.17% |
| NEED_LOGISTICS | `three` | 7,002 | 6,757 | 245 | 96.50% | 19.84% |
| **Total** | | **35,134** | **34,061** | **1,073** | **96.95%** | **100.00%** |

## Sumber dan detektor

- Manifest menggunakan 15 kolom dari Step 1 dan memilih 35,134 dari 35,134 gambar.
- Setiap baris membawa `image_id`, `image_path` relatif terhadap `hagrid_filtered/`, kedua label, `original_split`, `hf_split`, dan `user_id` dari sumber. `original_split=train_val` mencatat split gabungan di ZIP.
- Python 3.12.16, MediaPipe 0.10.14, Hand Landmarker FP16 v1 dalam mode IMAGE pada CPU. [Dokumentasi MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python)
- Detektor mencoba crop persegi di sekitar bounding box gestur, crop dengan margin lebih besar, dan gambar penuh. Padding hitam dihapus secara matematis saat koordinat dikembalikan ke raster sumber.
- Kandidat dipilih berdasarkan IoU dan fraksi landmark pada kotak gestur. Kotak tangan lain diperhitungkan. Handedness anotasi dapat memutus seri jika `leading_conf` memenuhi ambang. Parameter dan ambang tercatat di [feature_schema.json](feature_schema.json).
- `handedness_score` menyimpan keyakinan klasifikasi Left/Right. `detection_confidence` kosong karena hasil API gambar hanya mengekspos landmark, world landmarks, dan klasifikasi handedness. Ambang deteksi tersimpan dalam skema.

## Preprocessing fitur

- [landmarks_raw.csv](../ai/dataset/processed/landmarks_raw.csv) menyimpan x dan y sebagai fraksi lebar dan tinggi gambar asli. z merupakan estimasi kedalaman relatif pergelangan yang dinyatakan dalam satuan lebar gambar.
- [landmarks_normalized.csv](../ai/dataset/processed/landmarks_normalized.csv) menyimpan `x0, y0, z0, …, x20, y20, z20`. Translasi memakai pergelangan (0). Delta dikalikan dengan `(W, H, W)` lalu dibagi rata-rata jarak 3D dari pergelangan ke MCP (5, 9, 13, 17).
- Untuk tangan `Left`, komponen x dicerminkan menuju konvensi tangan kanan berdasarkan hasil MediaPipe. Label gestur hanya dipakai untuk memilih kotak anotasi. Vektor fitur hanya berisi koordinat.

## Pemeriksaan kualitas

- 34,061 pasangan raw/normalized lolos pemeriksaan 63 nilai berhingga, urutan fitur, normalisasi ulang, metadata dan sumber bounding box. Terdapat 0 temuan validasi.
- 7,882 gambar berhasil punya kotak tangan tambahan. Pada 124 gambar berhasil, percobaan yang dipakai mengembalikan lebih dari satu kandidat.
- Hasil menurut jalur deteksi mencakup `roi` (30,992), `expanded_roi` (2,544), `full_frame` (525).
- Metode pemilihan yang tercatat mencakup `target_bbox_geometry` (34,056), `target_bbox_geometry_handedness_tiebreak` (5).
- MediaPipe mengklasifikasikan 15,520 tangan sebagai Left dan 18,541 sebagai Right. Kesamaan dengan anotasi `leading_hand` 33,266/34,061 (97.67%).
- 10 gambar berhasil memiliki setidaknya satu x/y di luar batas raster, dengan selisih maksimum 0,0679 dalam koordinat gambar ternormalisasi. Nilai dipertahankan untuk audit dan normalisasi.
- Pemeriksaan visual mencakup 20 overlay, empat per kelas. Kerangka jatuh pada tangan gestur target pada contoh yang ditinjau, termasuk gambar dengan kotak tangan tambahan dan hasil fallback. Beberapa crop kecil tampak buram. [Sampel visual](../ai/dataset/processed/landmark_visual_samples/)

### Alasan kegagalan

- `no_hand_detected` — 932 (NEED_MEDICAL 340, NEED_LOGISTICS 212, EVACUATION 209, SAFE 115, DANGER_STOP 56).
- `no_target_bbox_match` — 140 (NEED_MEDICAL 49, EVACUATION 41, NEED_LOGISTICS 32, SAFE 15, DANGER_STOP 3).
- `ambiguous_target_hand` — 1 (NEED_LOGISTICS 1).

## Menjalankan ulang

Tiga CSV berukuran 120.5 MiB. Checkpoint SQLite di `ai/dataset/.landmark_cache/` menyimpan hasil inferensi per ID untuk melanjutkan proses dengan manifest, model, dan kode yang sama.

Jalankan dari akar repository dengan Python 3.12. Jika salinan hasil sudah tersedia, validasi dengan `--check-only`. Gunakan `--output` dengan direktori baru untuk menjalankan ekstraksi lain.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ai/requirements.txt
.venv/bin/python ai/extract_landmarks.py
.venv/bin/python ai/validate_landmarks.py
```

Sampel visual menggunakan gambar HaGRID 384p dan menambahkan overlay landmark. [Dataset sumber](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Lisensi HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)
