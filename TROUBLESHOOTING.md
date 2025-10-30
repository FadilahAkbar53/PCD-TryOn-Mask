# 🐛 Troubleshooting & Bug Fixes

## Common Issues & Solutions

### ✅ FIXED: JSON Serialization Error

**Error:**

```
TypeError: Object of type int32 is not JSON serializable
```

**Cause:** NumPy int32/int64 cannot be directly serialized to JSON.

**Fix Applied:** Convert NumPy integers to Python native int before JSON dump.

**File:** `pipelines/dataset.py`

**Changes:**

```python
# Before (BROKEN):
'label': r['label'],
'bbox': r['bbox']

# After (FIXED):
'label': int(r['label']),  # Convert numpy int to Python int
'bbox': tuple(int(x) for x in r['bbox'])  # Convert all bbox coords
```

---

## Other Potential Issues

### 1. Memory Error During Training

**Error:**

```
MemoryError: Unable to allocate array
```

**Solution:**

```powershell
# Reduce parameters
python app.py train --k 128 --max_desc 50000 --orb_features 300
```

---

### 2. No Module Named 'cv2'

**Error:**

```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**

```powershell
pip install opencv-python
```

---

### 3. No Module Named 'sklearn'

**Error:**

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**

```powershell
pip install scikit-learn
```

---

### 4. Haar Cascade Not Found

**Error:**

```
FileNotFoundError: Could not find cascade: haarcascade_frontalface_default.xml
```

**Solution:**

```powershell
python setup_project.py
```

---

### 5. No Faces Detected During Inference

**Problem:** `infer` or `webcam` command shows "Detected 0 faces"

**Solutions:**

**A. Check test image:**

- Ensure image has clear frontal face
- Good lighting (not too dark/bright)
- Face size > 30x30 pixels

**B. Adjust cascade sensitivity:**

Edit `pipelines/infer.py`, line ~40:

```python
# More sensitive (detect more faces, may have false positives)
scale_factor=1.05,  # default: 1.1
min_neighbors=3     # default: 5
```

---

### 6. Mask Misaligned

**Problem:** Mask position is wrong on face

**Solution:**

Edit `pipelines/overlay.py`, line ~46:

```python
def compute_mask_transform(self, face_box,
                           scale_width=1.1,      # Try 1.2 or 1.3
                           scale_height=0.45,    # Try 0.5 or 0.6
                           y_offset_ratio=0.5):  # Try 0.4 or 0.6
```

---

### 7. Low Accuracy (< 80%)

**Solutions:**

**A. Increase codebook size:**

```powershell
python app.py train --k 512
```

**B. Try RBF kernel:**

```powershell
python app.py train --k 256 --svm rbf
```

**C. More training data:**

- Add more face images to `data/faces/`
- Add more non-face images to `data/non_faces/`

---

### 8. Training Too Slow

**Solutions:**

**A. Reduce dataset:**

```powershell
python app.py train --k 128 --max_desc 50000
```

**B. Use fewer ORB features:**

```powershell
python app.py train --orb_features 300
```

---

### 9. Webcam Not Working

**Error:**

```
Failed to open video source
```

**Solutions:**

**A. Check camera ID:**

```powershell
# Try different camera IDs
python app.py webcam --camera 0
python app.py webcam --camera 1
```

**B. Check camera permissions:**

- Windows Settings → Privacy → Camera
- Enable camera access for Python

---

### 10. Model Files Not Found

**Error:**

```
FileNotFoundError: models/svm.pkl not found
```

**Solution:**

```powershell
# Train first to generate models
python app.py train
```

---

## Performance Optimization

### Speed Up Training

**1. Use MiniBatch K-Means (already implemented):**

- Default batch_size=1024 is optimal

**2. Reduce descriptor sampling:**

```powershell
python app.py train --max_desc 100000  # default: 200000
```

**3. Use Linear kernel (faster than RBF):**

```powershell
python app.py train --svm linear  # default
```

---

### Speed Up Inference

**1. Reduce ORB features (edit in code):**

`pipelines/features.py`, line ~18:

```python
self.orb = cv2.ORB_create(
    nfeatures=300,  # default: 500
    ...
)
```

**2. Increase NMS threshold:**

`pipelines/infer.py`, line ~85:

```python
nms_threshold: float = 0.5  # default: 0.3
```

---

## Debug Mode

### Enable Verbose Logging

Add this at the top of `app.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Known Limitations

### 1. Frontal Faces Only

- Haar cascade struggles with profile views
- **Solution:** Use multi-view cascades or dlib

### 2. Poor Lighting

- ORB detector sensitive to lighting
- **Solution:** Add histogram equalization preprocessing

### 3. Small Faces

- Minimum face size: 30x30 pixels
- **Solution:** Upscale image before detection

### 4. Occlusion

- Partially hidden faces may be missed
- **Solution:** Use landmark-based detection

---

## Emergency Reset

If everything breaks, reset to clean state:

```powershell
# 1. Delete generated files
Remove-Item -Recurse -Force models/*
Remove-Item -Recurse -Force reports/*
Remove-Item data/dataset_splits.json

# 2. Re-run setup
python setup_project.py

# 3. Fresh training
python app.py train --k 256
```

---

## Getting Help

1. **Check logs** - Look for ERROR/WARNING messages
2. **Read error messages** - Usually very descriptive
3. **Check file paths** - Windows uses backslashes
4. **Verify dependencies** - `pip list | grep -E "opencv|sklearn|numpy"`
5. **Check dataset** - `python check_dataset.py`

---

## Quick Diagnostics

Run this to check system health:

```powershell
# Check Python version
python --version  # Should be 3.10+

# Check dependencies
pip show opencv-python scikit-learn numpy

# Check dataset
python check_dataset.py

# Check models exist
dir models\

# Test imports
python -c "import cv2, sklearn, numpy; print('All OK!')"
```

---

## Bug Report Template

If you find a bug, report with:

```
**Error Message:**
[paste full traceback]

**Command Run:**
[paste command]

**Python Version:**
[python --version]

**System:**
[Windows 10/11, RAM, CPU]

**Dataset Size:**
[number of images]

**Steps to Reproduce:**
1. ...
2. ...
```

---

**Last Updated:** 2025-10-30
**Version:** 1.0.1 (JSON serialization fix)
