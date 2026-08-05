import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================
# PATHS
# ==========================

input_path = r"D:\pycharm_harini\Dataset"
output_path = r"D:\pycharm_harini\Processed_Dataset_v4"

os.makedirs(output_path, exist_ok=True)

# ==========================
# CLAHE
# ==========================

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ==========================
# CROP RETINA
# ==========================

def crop_retina(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = gray > 10

    coords = np.argwhere(mask)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img[y0:y1, x0:x1]

    return cropped


# ==========================
# CIRCULAR MASK
# ==========================

def circular_mask(img):

    h, w = img.shape[:2]

    mask = np.zeros((h,w), np.uint8)

    center = (w//2, h//2)

    radius = min(center[0], center[1])

    cv2.circle(mask, center, radius, 255, -1)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


# ==========================
# CLAHE
# ==========================

def apply_clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


# ==========================
# PROCESSING LOOP
# ==========================

for class_name in os.listdir(input_path):

    class_input = os.path.join(input_path, class_name)
    class_output = os.path.join(output_path, class_name)

    os.makedirs(class_output, exist_ok=True)

    images = os.listdir(class_input)

    print(f"\nProcessing class: {class_name}")

    for img_name in tqdm(images, desc=class_name):

        img_path = os.path.join(class_input, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # -------- pipeline --------

        img = crop_retina(img)

        img = circular_mask(img)

        img = cv2.resize(img, (256,256))

        img = apply_clahe(img)

        # save

        save_path = os.path.join(class_output, img_name)

        cv2.imwrite(save_path, img)

print("\n✅ Clean Preprocessing Completed")