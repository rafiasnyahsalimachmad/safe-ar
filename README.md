# S.A.F.E. AR — Kelompok 6

S.A.F.E. AR (Signal Assistance & Field Emergency AR) adalah proyek augmented reality untuk menyampaikan lima jenis sinyal bantuan melalui gestur tangan. Proyek Unity berada di `Assets/`, `Packages/`, dan `ProjectSettings/`. Versi editor yang tercatat di checkout ini adalah [Unity 6000.3.23f1](ProjectSettings/ProjectVersion.txt).

Pipeline Python di `ai/` menyiapkan gambar HaGRID dan mengubah gestur target menjadi vektor 63 fitur dari 21 landmark MediaPipe. Vektor ini merupakan masukan untuk tahap Random Forest berikutnya. **Step 1 dan 2 selesai. Step 3, pelatihan Random Forest, belum dilakukan.**

| Sinyal S.A.F.E. AR | Label HaGRID | Gestur |
| --- | --- | --- |
| `DANGER_STOP` | `palm` | Telapak terbuka dengan jari direntangkan |
| `NEED_MEDICAL` | `fist` | Tangan terkepal |
| `EVACUATION` | `one` | Satu telunjuk terangkat |
| `SAFE` | `ok` | Ibu jari dan telunjuk membentuk lingkaran |
| `NEED_LOGISTICS` | `three` | Tiga jari terangkat |

## Pipeline Python

| File | Fungsi |
| --- | --- |
| `ai/inspect_hagrid.py` | Memeriksa struktur ZIP, label, split, dan anotasi HaGRID |
| `ai/hagrid_source.py` | Membaca sumber ZIP lokal atau HTTP Range dan memeriksa anotasi |
| `ai/filter_hagrid.py` | Mengekspor lima kelas dan metadata sumber |
| `ai/validate_dataset.py` | Memvalidasi JPEG dan membuat laporan serta contact sheet |
| `ai/extract_landmarks.py` | Memilih tangan sesuai anotasi dan mengekspor landmark, fitur, serta status per gambar |
| `ai/feature_extractor.py` | Mengembalikan koordinat dari crop ke gambar sumber dan menormalkan fitur |
| `ai/validate_landmarks.py` | Memeriksa pasangan CSV, menghitung ulang fitur, dan membuat laporan serta overlay |

### Instalasi

Jalankan perintah dari akar clone `safe-ar` dengan Python 3.12. MediaPipe 0.10.14 telah diuji dengan Python 3.12.16 pada macOS Intel.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ai/requirements.txt
```

Di Windows, buat lingkungan dengan `py -3.12 -m venv .venv` dan aktifkan melalui `.venv\Scripts\Activate.ps1` di PowerShell. Perintah pipeline berikut sama setelah lingkungan aktif.

### Menghasilkan dataset dari sumber

Gunakan folder tujuan yang baru untuk eksperimen baru. Saat satu set ekspor sudah tersedia di lokasi default, script ekstraksi melewatinya. Step 1 dapat memakai `--archive /path/ke/hagrid-sample-120k-384p.zip` untuk membaca ZIP lokal. Step 2 memakai model Hand Landmarker yang disertakan di repository dan dapat mengambil bundel yang sama jika file lokal belum tersedia. [Model MediaPipe](ai/models/README.md)

```sh
python ai/inspect_hagrid.py
python ai/filter_hagrid.py
python ai/validate_dataset.py
python ai/extract_landmarks.py
python ai/validate_landmarks.py
```

Clone berisi empat CSV hasil Step 1 dan Step 2. Untuk memvalidasi vektor dan metadata CSV, jalankan perintah berikut dari akar repository.

```sh
python ai/validate_landmarks.py --check-only
```

Untuk bekerja dengan JPEG sumber, salin lima folder kelas dari workspace lama atau jalankan `python ai/filter_hagrid.py --repair` untuk mengambil gambar yang belum tersedia. Setelah JPEG tersedia, jalankan `python ai/validate_dataset.py --check-only` untuk memvalidasinya. Opsi `--check-only` mempertahankan laporan dan gambar pratinjau lokal.

`--output` memilih direktori hasil atau direktori yang divalidasi. `--source` pada kedua script landmark memilih folder yang berisi `metadata.csv` dan JPEG hasil Step 1.

## Dataset, lisensi, dan hasil

Sumber Step 1 adalah sampel HaGRID 384p oleh cj-mills. Dataset card mencantumkan CC BY-SA 4.0, dan repo HaGRID merujuk pada lisensi publik HaGRID tersendiri. Saat membagikan materi, cantumkan atribusi, pertahankan tautan lisensi, dan nyatakan perubahan sesuai ketentuan sumber. [Dataset card](https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p) · [Lisensi HaGRID](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)

Git menyimpan metadata **35.134 gambar**, CSV untuk **34.061 vektor valid**, log **1.073 kegagalan**, dan bundel Hand Landmarker. JPEG HaGRID, gambar pratinjau, serta cache pemrosesan tersedia di penyimpanan lokal. [Rincian dataset](ai/dataset/README.md)

- [Laporan Step 1](docs/dataset_report.md)
- [Laporan Step 2](docs/extraction_report.md)
- [Skema 63 fitur](docs/feature_schema.json)
