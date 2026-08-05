import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b2
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ========================
# DEVICE
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", device)

# ========================
# DATA PATH
# ========================

train_dir = r"D:\pycharm_harini\Processed_Split_train\train"
val_dir   = r"D:\pycharm_harini\Processed_Split_train\val"

# ========================
# TRANSFORMS
# ========================

train_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ========================
# DATASET
# ========================

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

classes = train_dataset.classes

print("📌 Classes:", classes)

# ========================
# CLASS WEIGHTS
# ========================

class_counts = np.bincount(train_dataset.targets)

weights = 1.0 / class_counts

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("Class Weights:", class_weights)

# ========================
# MODEL
# ========================

model = efficientnet_b2(weights="DEFAULT")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

model = model.to(device)

# ========================
# LOSS
# ========================

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ========================
# OPTIMIZER
# ========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ========================
# TRAINING LOOP
# ========================

epochs = 30
best_acc = 0

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    model.train()

    running_loss = 0

    for images,labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    print("Train Loss:", train_loss)

    # =================
    # VALIDATION
    # =================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,preds = torch.max(outputs,1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total

    print("Validation Accuracy:", val_acc)

    # =================
    # SAVE BEST MODEL
    # =================

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "dr_model_b2.pth"
        )

        print("✅ Best model saved")

print("\n🔥 Best Validation Accuracy:", best_acc)