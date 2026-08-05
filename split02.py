import os
import random
import shutil
from tqdm import tqdm

# input dataset
dataset_path = r"D:\Main_Prj\DATA_SET"

# output dataset
output_path = r"/Split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\nProcessing class: {class_name}")

    for split, img_list in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):

        split_folder = os.path.join(output_path, split, class_name)

        os.makedirs(split_folder, exist_ok=True)

        for img in tqdm(img_list, desc=f"{class_name} → {split}"):

            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)

            shutil.copy(src, dst)


print("\n✅ Dataset Split Completed")