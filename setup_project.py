"""
Setup script to prepare the project for first run.
Copies dataset and downloads Haar cascade files.
"""
import shutil
from pathlib import Path
import urllib.request
import cv2


def copy_dataset():
    """Copy dataset from parent folder to project folder."""
    print("Copying dataset...")
    
    source_faces = Path("../Dataset/faces")
    source_non_faces = Path("../Dataset/non_faces")
    
    dest_faces = Path("data/faces")
    dest_non_faces = Path("data/non_faces")
    
    if source_faces.exists() and source_non_faces.exists():
        # Copy faces
        if not dest_faces.exists():
            shutil.copytree(source_faces, dest_faces)
            print(f"✓ Copied faces from {source_faces}")
        else:
            print(f"✓ Faces already exist in {dest_faces}")
        
        # Copy non_faces
        if not dest_non_faces.exists():
            shutil.copytree(source_non_faces, dest_non_faces)
            print(f"✓ Copied non_faces from {source_non_faces}")
        else:
            print(f"✓ Non-faces already exist in {dest_non_faces}")
        
        # Count files
        face_count = len(list(dest_faces.rglob("*.jpg"))) + len(list(dest_faces.rglob("*.png")))
        non_face_count = len(list(dest_non_faces.rglob("*.jpg"))) + len(list(dest_non_faces.rglob("*.png")))
        
        print(f"\n✓ Dataset ready: {face_count} faces, {non_face_count} non-faces")
    else:
        print("✗ Source dataset not found in ../Dataset/")
        print("  Please ensure dataset is in the parent folder")


def download_cascades():
    """Download Haar cascade XML files from OpenCV."""
    print("\nChecking Haar cascade files...")
    
    cascade_dir = Path("assets/cascades")
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    cascades = {
        'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
    }
    
    for filename, url in cascades.items():
        dest_path = cascade_dir / filename
        
        if dest_path.exists():
            print(f"✓ {filename} already exists")
        else:
            try:
                # Try to get from OpenCV installation first
                cv2_data_dir = Path(cv2.__file__).parent / "data"
                cv2_cascade = cv2_data_dir / filename
                
                if cv2_cascade.exists():
                    shutil.copy(cv2_cascade, dest_path)
                    print(f"✓ Copied {filename} from OpenCV installation")
                else:
                    # Download from GitHub
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, dest_path)
                    print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to get {filename}: {e}")


def create_mask_placeholder():
    """Create a placeholder for mask.png if not exists."""
    print("\nChecking mask image...")
    
    mask_path = Path("assets/mask.png")
    
    if mask_path.exists():
        print(f"✓ Mask image already exists: {mask_path}")
    else:
        print(f"⚠ Mask image not found!")
        print(f"  Please place a transparent PNG mask at: {mask_path}")
        print(f"  You can download from: https://www.flaticon.com/search?word=mask")


def main():
    """Run all setup tasks."""
    print("="*60)
    print("SVM+ORB Face Detection - Project Setup")
    print("="*60)
    print()
    
    copy_dataset()
    download_cascades()
    create_mask_placeholder()
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place mask.png in assets/ folder (if not already done)")
    print("2. Run training: python app.py train --pos_dir data/faces --neg_dir data/non_faces")
    print("3. Enjoy! 🎉")


if __name__ == '__main__':
    main()
