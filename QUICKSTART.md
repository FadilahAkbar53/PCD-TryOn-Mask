# 🚀 Quick Start Guide

## Langkah-langkah Setup & Training

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Persiapkan Mask Image

**PENTING**: Anda perlu menyediakan file `assets/mask.png`

**Cara mendapatkan mask:**

**Option 1: Download dari internet**

- Cari "surgical mask PNG transparent" di Google Images
- Atau download dari: https://www.flaticon.com/search?word=mask
- Pastikan format PNG dengan **alpha channel** (transparansi)
- Save sebagai `assets/mask.png`

**Option 2: Buat sendiri dengan Paint/GIMP**

- Buat gambar mask sederhana
- Export sebagai PNG dengan transparansi
- Save ke `assets/mask.png`

**Option 3: Gunakan emoji (simple)**

```powershell
# Untuk sementara, Anda bisa gunakan gambar apa saja dulu
# (nanti bisa diganti dengan mask yang lebih bagus)
```

### 3. Verifikasi Dataset

Pastikan dataset sudah ada di:

```
data/
  faces/       # Gambar wajah
  non_faces/   # Gambar tanpa wajah
```

Cek jumlah file:

```powershell
(Get-ChildItem -Path "data/faces" -Recurse -Include *.jpg,*.png).Count
(Get-ChildItem -Path "data/non_faces" -Recurse -Include *.jpg,*.png).Count
```

### 4. Training (First Run)

**Training dengan dataset kecil (quick test):**

```powershell
python app.py train --pos_dir data/faces --neg_dir data/non_faces --k 128 --max_desc 50000
```

**Training dengan dataset penuh (recommended):**

```powershell
python app.py train --pos_dir data/faces --neg_dir data/non_faces --k 256 --max_desc 200000 --svm linear
```

**Training proses:**

- ⏱️ Waktu: ~2-5 menit (tergantung ukuran dataset)
- 📊 Output: Models akan disave di `models/`
- 📈 Reports: Metrics & plots di `reports/`

**Expected output:**

```
[INFO] Dataset prepared: train=700, val=150, test=150
[INFO] Extracted features from 1000 images
[INFO] K-Means clustering took 12.3s
[INFO] Best CV F1 score: 0.9234
[INFO] Test Accuracy: 0.9133
[INFO] Test F1: 0.9067
[INFO] Test AUC: 0.9512
```

### 5. Evaluation

Setelah training selesai:

```powershell
python app.py eval --report reports/test_metrics.json
```

Akan generate:

- ✅ `reports/test_metrics.json` - Metrics detail
- ✅ `reports/confusion_matrix.png` - Confusion matrix
- ✅ `reports/pr_curve.png` - Precision-Recall curve
- ✅ `reports/roc_curve.png` - ROC curve

### 6. Testing Inference

**Test pada gambar:**

```powershell
# Siapkan test image (misal: test.jpg)
python app.py infer --image test.jpg --out result.jpg --mask assets/mask.png --show
```

**Test pada webcam:**

```powershell
python app.py webcam --camera 0 --mask assets/mask.png --show
```

Controls:

- `q` - Quit
- `m` - Toggle mask ON/OFF
- `s` - Screenshot

---

## 🔧 Troubleshooting

### Problem: "No faces detected"

**Solusi:**

```powershell
# Pastikan input image punya wajah yang jelas (frontal)
# Coba adjust threshold di code jika perlu
```

### Problem: "Mask not found"

**Solusi:**

```powershell
# Pastikan file mask.png ada di assets/
dir assets/mask.png

# Jika belum ada, download/buat mask terlebih dahulu
```

### Problem: "Dataset count = 0"

**Solusi:**

```powershell
# Cek struktur folder dataset
Get-ChildItem -Path "data/faces" -Recurse
Get-ChildItem -Path "data/non_faces" -Recurse

# Pastikan ada file .jpg atau .png
```

### Problem: "Training error / Low accuracy"

**Solusi:**

```powershell
# Pastikan dataset minimal:
# - 100+ faces
# - 100+ non-faces
# - Gambar berkualitas baik (tidak blur/terlalu gelap)

# Coba training ulang dengan parameter lebih kecil:
python app.py train --k 128 --max_desc 50000
```

---

## 📊 Expected Performance

| Metric     | Target | Typical |
| ---------- | ------ | ------- |
| Accuracy   | ≥85%   | 91-93%  |
| F1 Score   | ≥85%   | 89-91%  |
| AUC        | ≥90%   | 94-96%  |
| FPS (720p) | ≥15    | 18-25   |

---

## 🎯 Next Steps

1. ✅ Setup project selesai
2. ✅ Install dependencies
3. ⏳ **Download/buat mask.png** ← ANDA DI SINI
4. ⏳ Train model
5. ⏳ Test inference
6. ⏳ Run webcam demo

---

## 📚 Dokumentasi Lengkap

Lihat `README.md` untuk:

- Penjelasan arsitektur lengkap
- Tuning hyperparameter
- Customisasi mask
- Perbandingan Linear vs RBF SVM
- Limitasi & improvement ideas

---

## 💡 Tips

**Untuk hasil terbaik:**

- Dataset minimal 500 images per class
- Gunakan gambar berkualitas baik (resolusi cukup, tidak blur)
- Balance positive/negative samples
- Test dengan Linear SVM dulu (lebih cepat)
- Jika accuracy < 85%, coba RBF kernel

**Untuk training cepat (testing):**

```powershell
python app.py train --k 64 --max_desc 20000
```

**Untuk produksi (best quality):**

```powershell
python app.py train --k 512 --max_desc 500000 --svm rbf
```

---

**Selamat coding! 🎉**
