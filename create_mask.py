"""
Create a simple mask placeholder for testing.
Replace this with a real transparent mask PNG later.
"""
import cv2
import numpy as np
from pathlib import Path


def create_simple_mask(output_path: str = 'assets/mask.png', width: int = 400, height: int = 200):
    """
    Create a simple rectangular mask with transparency.
    This is just a placeholder - replace with a real mask image!
    """
    # Create BGRA image (with alpha channel)
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Draw a rounded rectangle (mask shape)
    center_x = width // 2
    center_y = height // 2
    
    # Main mask area (white/light color)
    cv2.ellipse(mask, (center_x, center_y), (width//2 - 20, height//2 - 10), 
                0, 0, 360, (220, 220, 240, 230), -1)
    
    # Add some detail (simulating mask folds)
    for i in range(3):
        y = center_y - 30 + i * 30
        cv2.line(mask, (50, y), (width - 50, y), (200, 200, 220, 200), 2)
    
    # Add border
    cv2.ellipse(mask, (center_x, center_y), (width//2 - 20, height//2 - 10), 
                0, 0, 360, (180, 180, 200, 255), 3)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"✓ Created simple mask at: {output_path}")
    print(f"  Size: {width}x{height}")
    print(f"\n⚠️  This is a SIMPLE PLACEHOLDER mask!")
    print(f"  For better results, replace with a real mask PNG:")
    print(f"  - Download from: https://www.flaticon.com/search?word=surgical+mask")
    print(f"  - Or search: 'surgical mask PNG transparent' on Google Images")


if __name__ == '__main__':
    create_simple_mask()
