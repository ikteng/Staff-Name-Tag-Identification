import os
from pathlib import Path
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

def crop_rotated_bbox(image, center_x, center_y, width, height, rotation_deg):
    """
    Crop a rotated bounding box from an image
    """
    # Compute rotation matrix
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, 1.0)
    
    # Rotate the whole image
    rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    # Compute coordinates of the bbox in the rotated image
    x_min = int(center_x - width / 2)
    x_max = int(center_x + width / 2)
    y_min = int(center_y - height / 2)
    y_max = int(center_y + height / 2)
    
    # Ensure bounds
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, rotated.shape[1]), min(y_max, rotated.shape[0])
    
    return rotated[y_min:y_max, x_min:x_max]

def overlay_tag(image, tag_path='tag.png'):
    """
    Overlay a randomly augmented tag on the input image with transparency and 3D-like effect
    """
    img = image.copy()
    
    # Rotate if too tall
    if img.shape[0] / img.shape[1] <= 0.7:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    # Load the tag image
    tag_img = cv2.imread(tag_path, cv2.IMREAD_UNCHANGED)  # supports transparency
    if tag_img is None:
        raise FileNotFoundError(f"{tag_path} not found.")
    
    # Ensure tag has alpha channel
    if tag_img.shape[2] == 3:
        b, g, r = cv2.split(tag_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        tag_img = cv2.merge([b, g, r, alpha])
    
    # Apply blur
    blur_strength = (5, 5)
    tag_img[:, :, :3] = cv2.GaussianBlur(tag_img[:, :, :3], blur_strength, 0)
    
    # Subtle random darkening
    alpha_scale = random.uniform(0.85, 0.95)
    tag_img[:, :, :3] = (tag_img[:, :, :3] * alpha_scale).astype(np.uint8)
    
    # Resize tag
    scale_factor = random.uniform(0.08, 0.11)
    w = max(1, int(img.shape[1] * scale_factor))
    h = max(1, int(w * 0.6))
    tag_img = cv2.resize(tag_img, (w, h), interpolation=cv2.INTER_AREA)
    
    # 2D rotation with transparency
    rotate_val = random.choice([-20, -10, 0, 10, 20])
    tag_pil = Image.fromarray(tag_img)
    tag_rotated = tag_pil.rotate(rotate_val, expand=True, fillcolor=(0,0,0,0))  # transparent
    tag_img = np.array(tag_rotated)
    
    # 3D-like perspective
    h_t, w_t = tag_img.shape[:2]
    max_shift = w_t * 0.1
    pts1 = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]])
    pts2 = np.float32([
        [random.uniform(0, max_shift), random.uniform(0, max_shift)],
        [w_t - random.uniform(0, max_shift), random.uniform(0, max_shift)],
        [w_t - random.uniform(0, max_shift), h_t - random.uniform(0, max_shift)],
        [random.uniform(0, max_shift), h_t - random.uniform(0, max_shift)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    tag_img = cv2.warpPerspective(tag_img, M, (w_t, h_t), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # Random position
    height, width = img.shape[:2]
    x_min, x_max = width // 4, (width // 4) * 2
    y_min, y_max = height // 4, (height // 4) * 2
    x_rand = random.randint(x_min, int(x_max * 0.8))
    y_rand = random.randint(y_min, int(y_max * 0.8))
    
    # Overlay respecting alpha channel
    h_tag, w_tag = tag_img.shape[:2]
    end_y = min(y_rand + h_tag, height)
    end_x = min(x_rand + w_tag, width)
    overlay_h, overlay_w = end_y - y_rand, end_x - x_rand
    if overlay_h <=0 or overlay_w <=0:
        return img  # nothing to overlay

    overlay = tag_img[:overlay_h, :overlay_w]
    alpha = overlay[:, :, 3:] / 255.0
    img[y_rand:end_y, x_rand:end_x, :3] = (alpha * overlay[:, :, :3] + (1-alpha) * img[y_rand:end_y, x_rand:end_x, :3]).astype(np.uint8)
    
    return img

def compress(image, path):
    """
    Compress and degrade an image to simulate poor quality.
    Combines low JPEG quality, slight blur, and noise.
    """
    # Save as low-quality JPEG first
    quality = random.randint(10, 40)  # aggressive low-quality
    img_pil = Image.fromarray(image)
    img_pil.save(path, "JPEG", quality=quality, optimize=True)
    
    # Reload image and apply slight blur
    img = np.array(Image.open(path))
    blur_size = random.choice([3,5])
    img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    # Add small random noise
    noise = np.random.normal(0, random.randint(5, 15), img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Save again
    Image.fromarray(img).save(path, "JPEG", quality=quality, optimize=True)

def generate_dataset_images(dataset_splits: list[list[str]], base_filename: str):
    """
    Crop objects from images using annotations, apply augmentation, 
    and save positive and negative samples for train/test/val splits.
    """
    global_counter = 0

    print(f"\nProcessing dataset for '{base_filename}'...")

    split_names = ['train', 'test', 'val']
    for split_idx, image_paths in enumerate(dataset_splits):
        split_name = split_names[split_idx]
        print(f"Generating {split_name} set with {len(image_paths)} images...")

        # Ensure directories exist once per split
        for label in ['0', '1']:
            Path(f"processed_dataset/{split_name}/{label}").mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {split_name}", ncols=100)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Could not read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load annotation
            ann_path = Path(img_path).with_suffix('.txt')
            if not ann_path.exists():
                print(f"  Missing annotation file: {ann_path}")
                continue

            # Read bounding boxes
            bboxes = []
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # skip malformed lines
                    # Skip class ID, keep only bbox coordinates
                    bboxes.append([float(x) for x in parts[1:]])

            if not bboxes:
                print(f"  No bounding boxes in: {ann_path}")
                continue

            # Crop and augment each object
            for obj_idx, bbox in enumerate(bboxes):
                cropped_obj = crop_rotated_bbox(img, *bbox)

                # Generate augmented negative version
                neg_obj = overlay_tag(cropped_obj)

                # Generate unique filename
                filename = f"{base_filename}_{obj_idx}_{global_counter}.jpg"

                pos_path = os.path.join("processed_dataset", split_name, "0", filename)
                neg_path = os.path.join("processed_dataset", split_name, "1", filename)

                # Save images
                compress(cropped_obj, pos_path)
                compress(neg_obj, neg_path)

                global_counter += 1

        print(f"  ✅ Completed {split_name} set for '{base_filename}' ({global_counter} images total so far)")

def split_dataset(data):
    """
    Splits data into train (70%), validation (15%), and test (15%)
    """
    train, temp = train_test_split(data, test_size=0.30, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)
    
    return [train, val, test]

if __name__ == "__main__":
    HABBOF_path = 'HABBOF'
    folders = ["Lab1", "Lab2", "Meeting1", "Meeting2"]

    for folder in folders:
        folder_path = os.path.join(HABBOF_path, folder)
        
        # Get all jpg files in the folder
        data_arr = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        
        # Generate dataset images
        generate_dataset_images(split_dataset(data_arr), folder)

    print("Completed processing the dataset!")
