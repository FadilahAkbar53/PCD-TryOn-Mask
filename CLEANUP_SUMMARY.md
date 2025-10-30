# 🧹 CLEANUP SUMMARY

## ✅ File yang DI-KEEP (Commit ke Git)

### 📄 **Source Code**

```
app.py                      - Main CLI application
pipelines/
├── __init__.py            - Package init
├── dataset.py             - Dataset loader & ROI extractor
├── features.py            - ORB + BoVW feature extraction
├── train.py               - SVM trainer with GridSearch
├── infer.py               - Image/video inference
├── overlay.py             - Mask overlay engine
└── utils.py               - Utilities (NMS, plotting, etc.)
```

### 🛠️ **Utilities**

```
check_dataset.py           - Dataset verification tool
create_mask.py             - Mask generator
setup_project.py           - One-time setup script
```

### 📚 **Documentation**

```
README.md                  - Main documentation (600+ lines)
QUICKSTART.md              - Quick start guide
PROJECT_SUMMARY.md         - Project overview
STATUS.md                  - Development status
TROUBLESHOOTING.md         - Common issues & solutions
LICENSE                    - MIT License
```

### 📦 **Configuration**

```
requirements.txt           - Python dependencies
.gitignore                 - Git ignore rules
```

### 📓 **Notebooks**

```
notebooks/EDA.ipynb        - Exploratory data analysis
```

### 🖼️ **Assets**

```
assets/mask.png            - Default mask template
assets/cascades/           - Haar cascade XML files
```

---

## ❌ File yang DI-IGNORE (TIDAK commit ke Git)

### 🤖 **Trained Models** (LARGE - regenerate lokal)

```
models/codebook.pkl        - K-Means codebook (256 clusters)
models/scaler.pkl          - StandardScaler
models/svm.pkl             - Trained SVM classifier
```

**Alasan**: File besar (~10-50 MB), bisa di-regenerate dengan `python app.py train`

---

### 💾 **Dataset** (LARGE - user provide sendiri)

```
data/faces/                - 1,331 face images
data/non_faces/            - 1,317 non-face images
data/dataset_splits.json   - Train/val/test split info
```

**Alasan**: File sangat besar (~500 MB), user harus sediakan dataset sendiri

---

### 📊 **Reports** (Reproducible)

```
reports/confusion_matrix.png
reports/pr_curve.png
reports/roc_curve.png
reports/metrics.json
reports/test_metrics.json
```

**Alasan**: Bisa di-regenerate dengan `python app.py eval`

---

### 🗑️ **Temporary/Cache Files**

```
__pycache__/               - Python bytecode cache
*.pyc, *.pyo              - Compiled Python files
*.log                     - Log files
screenshot_*.jpg          - Webcam screenshots
output*.jpg               - Inference outputs
```

**Alasan**: Temporary files, auto-generated

---

### 🖥️ **IDE/OS Files**

```
.vscode/                  - VS Code settings
.idea/                    - PyCharm settings
.DS_Store                 - macOS metadata
Thumbs.db                 - Windows thumbnails
```

**Alasan**: IDE-specific, tidak perlu di-share

---

## 📏 **Ukuran Estimasi**

| Category            | Size (Approx) | Git Status    |
| ------------------- | ------------- | ------------- |
| Source Code         | ~100 KB       | ✅ **COMMIT** |
| Documentation       | ~50 KB        | ✅ **COMMIT** |
| Assets (cascades)   | ~1 MB         | ✅ **COMMIT** |
| **Models**          | ~30 MB        | ❌ **IGNORE** |
| **Dataset**         | ~500 MB       | ❌ **IGNORE** |
| **Reports**         | ~2 MB         | ❌ **IGNORE** |
| **Total to commit** | **~1.2 MB**   | ✅            |

---

## 🚀 **Cara Menggunakan di Git**

### **1. First Time Setup:**

```bash
git init
git add .
git commit -m "Initial commit: SVM+ORB face detector with mask overlay"
git remote add origin <your-repo-url>
git push -u origin main
```

### **2. Clone di Komputer Lain:**

```bash
git clone <your-repo-url>
cd svm_orb_mask
pip install -r requirements.txt
python setup_project.py  # Download cascades, setup folders
# Copy dataset ke data/faces dan data/non_faces
python app.py train --k 256 --max_desc 200000  # Train model
python app.py webcam --camera 0 --mask assets/mask.png  # Run!
```

---

## 📝 **Notes**

- Models dan dataset **TIDAK** di-commit karena ukuran besar
- User harus **train model sendiri** setelah clone
- User harus **sediakan dataset** sendiri (atau download dari link terpisah)
- Semua **source code dan dokumentasi** lengkap di-commit
- **Reproducible**: Bisa re-create semua hasil dari source code

---

**Last Updated**: October 30, 2025  
**Total Files Committed**: ~20 files (~1.2 MB)  
**Total Files Ignored**: ~2,700+ files (~530 MB)
