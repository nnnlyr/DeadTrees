import os
import random
import shutil

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, jaccard_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Improve efficiency
def classBalance(image, mask, window_size=32, stride=32, neg_ratio=2):
    pos_patches = []
    neg_patches = []

    h, w, c = image.shape

    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patch = image[i:i + window_size, j:j + window_size, :]
            label = mask[i + window_size // 2, j + window_size // 2]

            if patch.shape[:2] != (window_size, window_size):
                continue

            if label == 1:
                pos_patches.append((patch.flatten(), 1))
            elif label == 0:
                neg_patches.append((patch.flatten(), 0))

    neg_sampled = random.sample(
        neg_patches,
        k=min(len(neg_patches), len(pos_patches) * neg_ratio)
    )

    all_patches = pos_patches + neg_sampled
    X_patch = [p[0] for p in all_patches]
    y_patch = [p[1] for p in all_patches]

    return np.array(X_patch), np.array(y_patch)

def getTestPatches(image, mask, window_size=32, stride=32):
    h, w, c = image.shape
    X_patch = []
    y_patch = []
    coords = []
    mask_test = []

    mask_test = mask.flatten()

    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patch = image[i:i + window_size, j:j + window_size, :]
            if patch.shape[:2] != (window_size, window_size):
                continue

            center_i = i + window_size // 2
            center_j = j + window_size // 2
            label = mask[center_i, center_j]  # 二值图：0/1

            X_patch.append(patch.flatten())
            y_patch.append(label)
            coords.append((center_i, center_j))

    return np.array(X_patch), np.array(y_patch), coords, mask_test

# Import Dataset
img_root = 'data/NRG_images'
mask_root = 'data/masks'

all_files = []

for file in os.listdir(img_root):
    nrg_path = os.path.join(img_root, file)
    mask_file = file.replace("NRG", "mask")
    mask_path = os.path.join(mask_root, mask_file)

    if os.path.exists(mask_path):
        all_files.append((mask_file, file))

# Split into train set and test set
train_files, test_files = train_test_split(all_files, test_size = 0.2, random_state = 42)

X_train = []
y_train = []

# Load and split the images in train set
for mask_file, file in train_files:
    mask = cv2.resize(cv2.imread(os.path.join(mask_root, mask_file), cv2.IMREAD_GRAYSCALE), (128, 128), interpolation = cv2.INTER_NEAREST)
    mask = (mask > 128).astype(np.uint8)
    nrg_img = cv2.resize(cv2.imread(os.path.join(img_root, file), cv2.IMREAD_UNCHANGED), (128, 128), interpolation = cv2.INTER_AREA)
    X_patch, y_patch = classBalance(nrg_img, mask)

    if len(X_patch) == 0:
        continue

    X_train.append(X_patch)
    y_train.append(y_patch)

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

X_test = []
X_test_final = []
y_test = []
coords = []
mask_test = []

# Load and split the images in test set
for mask_file, file in test_files:
    mask = cv2.resize(cv2.imread(os.path.join(mask_root, mask_file), cv2.IMREAD_GRAYSCALE), (128, 128), interpolation = cv2.INTER_NEAREST)
    mask = (mask > 128).astype(np.uint8)
    nrg_img = cv2.resize(cv2.imread(os.path.join(img_root, file), cv2.IMREAD_UNCHANGED), (128, 128), interpolation = cv2.INTER_AREA)
    X_patch, y_patch, coord, masks = getTestPatches(nrg_img, mask)
    X_test.append(X_patch)
    X_test_final.append(X_patch)
    y_test.append(y_patch)
    coords.append(coord)
    mask_test.append(masks)

X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

# Random Forests
tree_count = list(range(10, 100, 10))
random_forest_accuracy = []
best_accuracy = 0

for count in tree_count:
    random_tree_model = RandomForestClassifier(n_estimators = count, random_state = 10, max_depth = 15, n_jobs=-1, class_weight='balanced')
    random_tree_model.fit(X_train, y_train)
    y_pred_rf = random_tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    random_forest_accuracy.append(accuracy)
    if accuracy >= best_accuracy:
        best_count = count

random_tree_model = RandomForestClassifier(n_estimators = best_count, random_state = 10, max_depth = 15, n_jobs=-1, class_weight='balanced')
random_tree_model.fit(X_train, y_train)

y_test_final = []
y_pred = []

for X_img, coords_img, mask_flat in zip(X_test_final, coords, mask_test):
    y_pred_img = random_tree_model.predict(X_img)

    pred_map = np.zeros((128, 128), dtype=np.uint8)
    for (i, j), pred in zip(coords_img, y_pred_img):
        pred_map[i, j] = pred

    y_test_final.extend(mask_flat)
    y_pred.extend(pred_map.flatten())

accuracy = accuracy_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
recall = recall_score(y_test_final, y_pred)
iou = jaccard_score(y_test_final, y_pred)

# Outcome output
print('-----------Random Forest----------------')
print(f'IoU: {iou: .4f}')
print(f'Accuracy: {accuracy: .4f}')
print(f'Precision: {precision: .4f}')
print(f'Recall: {recall: .4f}')
print(f'F1 Score: {f1: .4f}')
print(f'Confusion Matrix:')
print(confusion_matrix(y_true = y_test_final, y_pred = y_pred))
