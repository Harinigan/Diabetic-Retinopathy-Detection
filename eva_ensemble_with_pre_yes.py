import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b2, efficientnet_b3
from torchvision.transforms.functional import rotate
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    classification_report, accuracy_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", device)

# ========================
# DATA PATH
# ========================
test_dir = r"D:\Main_Prj\Split\test"

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========================
# DATASET
# ========================
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

print("📌 Classes:", test_dataset.classes)
print("📌 Total TEST images:", len(test_dataset))

NUM_CLASSES = len(test_dataset.classes)
CLASS_NAMES = test_dataset.classes

# ========================
# MODEL LOADER
# ========================
def load_model(model_fn, model_path: str, num_classes: int):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    model = model_fn(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model

model_b2 = load_model(efficientnet_b2, "dr_model_b2.pth",  NUM_CLASSES)
print("✅ EfficientNet-B2 Loaded")

model_b3 = load_model(efficientnet_b3, "best_dr_b3.pth", NUM_CLASSES)
print("✅ EfficientNet-B3 Loaded")

# ========================
# TTA — fundus-safe only
# ========================
def tta_predict(models: list, images: torch.Tensor) -> torch.Tensor:
    softmax = nn.Softmax(dim=1)
    augmented = [
        images,
        torch.flip(images, [3]),
        rotate(images,  10),
        rotate(images, -10),
        rotate(images,  20),
        rotate(images, -20),
    ]
    all_probs = torch.stack([
        softmax(model(aug))
        for model in models
        for aug in augmented
    ])
    return all_probs.mean(dim=0)

# ========================
# COLLECT PROBABILITIES
# ========================
all_probs_list = []
all_labels     = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Collecting probs"):
        images = images.to(device)
        probs  = tta_predict([model_b2, model_b3], images)
        all_probs_list.append(probs.cpu())
        all_labels.extend(labels.numpy())

all_probs  = torch.cat(all_probs_list, dim=0).numpy()
all_labels = np.array(all_labels)

# ========================
# THRESHOLD CALIBRATION
# class order: ['Mild','Moderate','NO_DR','Proliferative','Severe']
# ========================
class_weights = np.array([
    1.10,   # Mild
    1.00,   # Moderate
    1.00,   # NO_DR
    1.60,   # Proliferative
    1.20,   # Severe
], dtype=np.float32)

weighted_probs = all_probs * class_weights
all_preds      = np.argmax(weighted_probs, axis=1)

# ========================
# METRICS
# ========================
accuracy  = 100 * (all_preds == all_labels).mean()
recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)

# AUC per class (one-vs-rest on raw probs, not weighted)
labels_bin    = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
auc_per_class = []
for i in range(NUM_CLASSES):
    try:
        auc_val = roc_auc_score(labels_bin[:, i], all_probs[:, i])
    except Exception:
        auc_val = float('nan')
    auc_per_class.append(auc_val)
macro_auc = np.nanmean(auc_per_class)

print(f"\n🎯 Accuracy  : {accuracy:.2f}%")
print(f"📊 Recall    : {recall    * 100:.2f}%")
print(f"📊 PPV       : {precision * 100:.2f}%")
print(f"📊 Macro AUC : {macro_auc:.4f}")

print("\n📋 Per-class Report:")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES, zero_division=0))

print("\nPer-class AUC:")
for i, cname in enumerate(CLASS_NAMES):
    print(f"  {cname:<20} AUC = {auc_per_class[i]:.4f}")

# ========================
# FIGURE 1 — CONFUSION MATRIX
# ========================
cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

plt.figure(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — Ensemble (B2+B3) Enhanced")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("💾 Saved → confusion_matrix.png")

# ========================
# FIGURE 2 — ROC + AUC CURVE
# ========================
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

fig, ax = plt.subplots(figsize=(9, 7))

for i, (cname, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{cname}  (AUC = {roc_auc:.4f})")

# Macro-average ROC
all_fpr  = np.unique(np.concatenate([
    roc_curve(labels_bin[:, i], all_probs[:, i])[0]
    for i in range(NUM_CLASSES)
]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(NUM_CLASSES):
    fpr_i, tpr_i, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
mean_tpr     /= NUM_CLASSES
macro_auc_plot = auc(all_fpr, mean_tpr)

ax.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
        label=f"Macro-avg  (AUC = {macro_auc_plot:.4f})")
ax.plot([0, 1], [0, 1], 'k:', lw=1.2, label='Random classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
ax.set_title('ROC Curves — Ensemble (B2 + B3) Enhanced\nOne-vs-Rest per DR Grade', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_enhanced.png", dpi=150)
plt.show()
print("💾 Saved → roc_curve_enhanced.png")

# ========================
# FIGURE 3 — RECALL & PPV BAR CHART
# ========================
recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
ppv_per_class    = precision_score(all_labels, all_preds, average=None, zero_division=0)

x     = np.arange(NUM_CLASSES)
width = 0.35
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars1 = ax3.bar(x - width/2, recall_per_class * 100, width,
                label='Recall (Sensitivity)', color='steelblue', edgecolor='white')
bars2 = ax3.bar(x + width/2, ppv_per_class * 100, width,
                label='PPV (Precision)', color='coral', edgecolor='white')
ax3.set_xticks(x)
ax3.set_xticklabels(CLASS_NAMES, fontsize=11)
ax3.set_ylabel('Score (%)', fontsize=12)
ax3.set_ylim(0, 115)
ax3.set_title('Per-Class Recall & PPV — Ensemble (B2+B3) Enhanced', fontsize=13)
ax3.axhline(y=accuracy, color='green', linestyle='--', lw=1.5,
            label=f'Overall Accuracy ({accuracy:.1f}%)')
ax3.legend(fontsize=10)
for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("recall_ppv_bar_enhanced.png", dpi=150)
plt.show()
print("💾 Saved → recall_ppv_bar_enhanced.png")
print("\n🎉 All evaluation complete!")