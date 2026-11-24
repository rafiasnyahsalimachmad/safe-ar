# S.A.F.E. AR — Step 1. Inspeksi dan ekstraksi HaGRID

CSV tersedia dalam repository. Tautan gambar pratinjau mengarah ke artefak lokal.

**Validasi otomatis lulus**. 35,134 JPEG tervalidasi di `ai/dataset/hagrid_filtered/`.

## Pemetaan gestur dan jumlah JPEG

| Kelas S.A.F.E. AR | HaGRID | Kecocokan visual | Folder sumber | Lolos validasi | Ukuran JPEG (byte) |
|---|---|---|---:|---:|---:|
| DANGER_STOP | `palm` | Telapak menghadap kamera, lima jari terbuka | 7,082 | 7,082 | 182,512,435 |
| NEED_MEDICAL | `fist` | Tangan terkepal | 6,941 | 6,940 | 175,950,011 |
| EVACUATION | `one` | Satu telunjuk terangkat | 7,111 | 7,111 | 182,026,529 |
| SAFE | `ok` | Ibu jari dan telunjuk membentuk lingkaran | 6,999 | 6,999 | 181,099,565 |
| NEED_LOGISTICS | `three` | Telunjuk, jari tengah, dan jari manis terangkat | 7,004 | 7,002 | 178,433,877 |
| **Total** | | | | **35,134** | **900,022,417** |

Contoh sumber memperlihatkan `palm` dengan jari terbuka dan `stop` dengan jari rapat. Pemetaan DANGER_STOP memakai `palm`.
Contact sheet menampilkan sampai empat foto dan detail kotak tangan target per kelas. UUID di bawah foto mengacu pada `image_id` di `metadata.csv`.

## Sumber, split, dan anotasi

- Revisi Hugging Face `61d5663f48cf742257c0ec3666a5bda8a19bc86f`.
- Seluruh label yang ditemukan pada anotasi (19): `call`, `dislike`, `fist`, `four`, `like`, `mute`, `no_gesture`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`.
- ZIP berisi 127,331 JPEG dalam 18 folder kelas. `no_gesture` muncul pada 28,150 kotak tangan dan tidak memiliki folder gambar tersendiri.
- ZIP menyediakan 127,331 baris dalam 1 row group Parquet serta 18 berkas JSON kelas di `ann_train_val/`. Ekstraksi memakai Parquet dari dalam ZIP dan mencocokkan UUID gambar dengan indeks anotasi.
- `original_split=train_val` mengikuti nama folder ZIP. `hf_split=train` merekam nama split di viewer Hugging Face. Sampel ini menggabungkan train dan val tanpa tanda per gambar serta tidak menyediakan split test.
- `bboxes` memuat `[x, y, lebar, tinggi]` dengan titik asal kiri atas dan koordinat ternormalisasi. `labels` berurutan sesuai kotak. CSV juga menyimpan `leading_hand`, `leading_conf`, `user_id`, anotasi lengkap, path sumber, ukuran, dan CRC-32 setiap JPEG.
- Hasil memuat 43,383 kotak tangan, termasuk 8,249 berlabel `no_gesture`.
- Dataset card dan API viewer hanya menyediakan pratinjau. Endpoint Parquet publik tidak mengembalikan berkas dan filter viewer dinonaktifkan. Ekstraksi jaringan memakai HTTP Range pada ZIP dan menyimpan anggota JPEG terpilih tanpa pengodean ulang.
- [Dataset card](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Inventaris ZIP](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p/tree/61d5663f48cf742257c0ec3666a5bda8a19bc86f) · [HaGRID v1](https://raw.githubusercontent.com/hukenovs/hagrid/Hagrid_v1/README.md)

## Validasi dan kendala

- 35,134 JPEG lolos dekode, CRC-32, dan pemeriksaan ukuran. Dimensi asli yang ditemukan mencakup 291 ukuran; yang paling sering adalah 384×512 (21,526), 512×384 (3,695), 683×384 (1,616), 384×683 (1,549).
- Duplikat persis: 0 grup byte identik dan 0 grup piksel RGB identik. Pencarian duplikat memakai SHA-256 dan mencakup seluruh kelas.
- Path, ID, label, split, jumlah kotak, rentang koordinat, dan kolom anotasi diperiksa pada setiap baris.
- Anotasi juga dicocokkan dengan Parquet sumber berdasarkan UUID (35,134 gambar lolos).
- Pemeriksaan visual mencakup 20 contoh, empat per kelas. Pose target tampak pada kelima kelas. Beberapa tangan kecil atau buram karena jarak dan resolusi sumber. [Contact sheet](../ai/dataset/hagrid_filtered/contact_sheet.jpg)
- Di seluruh ZIP, 4 anotasi tidak memuat label folder dan 2 kotak berada di luar batas yang ditoleransi.
- 3 gambar dari folder target dikecualikan karena anotasinya tidak valid untuk kelas tersebut.
  - `hagrid-sample-120k-384p/hagrid_120k/train_val_fist/70970c12-ebf8-4021-8e85-6f9f70f9b829.jpg` (target_label_missing)
  - `hagrid-sample-120k-384p/hagrid_120k/train_val_three/47c4529a-f2e3-47c0-8c34-dedcddd293fd.jpg` (target_label_missing)
  - `hagrid-sample-120k-384p/hagrid_120k/train_val_three/47fe38b8-bae1-472d-8d0e-769222eef1b5.jpg` (target_label_missing)

## Ukuran dan lokasi

- JPEG: 900,022,417 byte (858.3 MiB).
- [metadata.csv](../ai/dataset/hagrid_filtered/metadata.csv): 21,791,715 byte. [Contact sheet](../ai/dataset/hagrid_filtered/contact_sheet.jpg): 358,715 byte.
- Total JPEG, CSV, dan contact sheet: **922,172,847 byte (879.5 MiB)**.

## Lisensi dan atribusi

- Dataset card mencantumkan CC BY-SA 4.0. Saat membagikan materi, sertakan atribusi dan tautan lisensi, nyatakan perubahan, serta gunakan lisensi yang sama atau kompatibel untuk adaptasi yang dibagikan. Lisensi standar juga melarang pembatasan tambahan atas hak yang diberikannya. [Dataset card](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Repositori asli HaGRID merujuk ke lisensi publik tersendiri yang diadaptasi dari CC BY-SA 4.0. Dokumen tersebut mengizinkan reproduksi dan adaptasi, mewajibkan atribusi dan pemberitahuan perubahan saat dibagikan, serta mensyaratkan lisensi dengan unsur setara atau kompatibel untuk adaptasi yang dibagikan. Hak privasi dan hak atas penggambaran orang berada di luar cakupan pemberian lisensinya. [Lisensi spesifik HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)
- Kredit HaGRID kepada Alexander Kapitanov, Andrey Makhlyarchuk, dan Karina Kvanchiani. cj-mills membuat sampel ukuran 384p. S.A.F.E. AR memilih lima kelas, memetakan label, mengekspor anotasi, dan membuat contact sheet. [Dataset card](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Publikasi HaGRID](https://arxiv.org/abs/2206.08219)

## Menjalankan ulang

Uji ulang pada ekspor ini melewati semua 35.134 JPEG dan mentransfer 0 byte melalui HTTP Range. Pada salinan uji, validator mendeteksi JPEG rusak, kotak di luar batas, dan duplikat lintas kelas. Ekstraksi ulang memperbaiki JPEG rusak serta melewati JPEG yang masih cocok.

Ekstraksi awal diuji dengan Python 3.13. Untuk menjalankan pipeline terintegrasi, gunakan Python 3.12 dari akar repository.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ai/requirements.txt
.venv/bin/python ai/inspect_hagrid.py
.venv/bin/python ai/filter_hagrid.py
.venv/bin/python ai/validate_dataset.py
```

Jika metadata ekspor sudah tersedia, filter melewati ekspor tersebut. Gunakan pemeriksaan `--check-only` untuk memvalidasi hasil yang ada. Opsi `--repair` memeriksa ulang ukuran dan CRC-32 JPEG serta mengambil kembali hanya berkas yang perlu dipulihkan. Gunakan `--archive /path/ke/sumber.zip` pada inspeksi dan ekstraksi untuk membaca ZIP yang sudah tersedia secara lokal. Validasi berjalan tanpa jaringan.
