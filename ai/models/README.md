# Model AI

Step 2 menggunakan bundel resmi Google MediaPipe Hand Landmarker FP16 v1. `hand_landmarker.task` disertakan di folder ini. `extract_landmarks.py` memverifikasi checksum dan mengambil bundel yang sama jika file belum tersedia. Untuk memakai salinan lokal di tempat lain, berikan path melalui `--model`.

SHA-256 untuk bundel ini adalah `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. Bundel Google ini didistribusikan tanpa perubahan sesuai [Apache License 2.0](LICENSE-MediaPipe.txt). [Kartu model](https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full) · [Bundel model resmi](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) · [Dokumentasi Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
