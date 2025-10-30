"""
Check dataset status and provide statistics.
"""
from pathlib import Path
from collections import Counter


def check_dataset():
    """Check dataset folders and count files."""
    print("="*60)
    print("DATASET STATUS CHECK")
    print("="*60)
    print()
    
    # Define paths
    faces_dir = Path("data/faces")
    non_faces_dir = Path("data/non_faces")
    
    # Check faces
    print("📁 Positive Samples (Faces):")
    print(f"   Path: {faces_dir}")
    
    if faces_dir.exists():
        face_files = list(faces_dir.rglob("*.jpg")) + list(faces_dir.rglob("*.png")) + \
                     list(faces_dir.rglob("*.jpeg")) + list(faces_dir.rglob("*.bmp"))
        print(f"   Status: ✅ Found")
        print(f"   Count: {len(face_files)} images")
        
        if len(face_files) > 0:
            # Show file extensions
            exts = Counter([f.suffix for f in face_files])
            print(f"   Formats: {dict(exts)}")
            
            # Sample files
            print(f"   Sample files:")
            for f in face_files[:3]:
                print(f"     - {f.name}")
        else:
            print(f"   ⚠️  WARNING: No image files found!")
            print(f"   Please add images to: {faces_dir}")
    else:
        print(f"   Status: ❌ NOT FOUND")
        print(f"   Please create: {faces_dir}")
    
    print()
    
    # Check non-faces
    print("📁 Negative Samples (Non-Faces):")
    print(f"   Path: {non_faces_dir}")
    
    if non_faces_dir.exists():
        non_face_files = list(non_faces_dir.rglob("*.jpg")) + list(non_faces_dir.rglob("*.png")) + \
                         list(non_faces_dir.rglob("*.jpeg")) + list(non_faces_dir.rglob("*.bmp"))
        print(f"   Status: ✅ Found")
        print(f"   Count: {len(non_face_files)} images")
        
        if len(non_face_files) > 0:
            # Show file extensions
            exts = Counter([f.suffix for f in non_face_files])
            print(f"   Formats: {dict(exts)}")
            
            # Sample files
            print(f"   Sample files:")
            for f in non_face_files[:3]:
                print(f"     - {f.name}")
        else:
            print(f"   ⚠️  WARNING: No image files found!")
            print(f"   Please add images to: {non_faces_dir}")
    else:
        print(f"   Status: ❌ NOT FOUND")
        print(f"   Please create: {non_faces_dir}")
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    if faces_dir.exists() and non_faces_dir.exists():
        total = len(face_files) + len(non_face_files)
        print(f"Total images: {total}")
        
        if total >= 200:
            print(f"✅ Dataset sufficient for training!")
            print(f"   Recommended: ≥200 images total")
            print(f"   Your dataset: {total} images")
        elif total >= 50:
            print(f"⚠️  Dataset small but usable")
            print(f"   Minimum: 50 images (25 per class)")
            print(f"   Your dataset: {total} images")
            print(f"   Recommendation: Add more images for better accuracy")
        else:
            print(f"❌ Dataset too small!")
            print(f"   Minimum: 50 images (25 per class)")
            print(f"   Your dataset: {total} images")
            print(f"   Please add more images before training")
        
        print()
        print("Next steps:")
        if total >= 50:
            print("1. Create mask: python create_mask.py")
            print("2. Start training: python app.py train")
        else:
            print("1. Add more images to data/faces/ and data/non_faces/")
            print("2. Run this check again: python check_dataset.py")
    else:
        print("❌ Dataset folders not ready")
        print("\nPlease ensure:")
        print("1. Folder data/faces/ exists with face images")
        print("2. Folder data/non_faces/ exists with non-face images")
    
    print()


if __name__ == '__main__':
    check_dataset()
