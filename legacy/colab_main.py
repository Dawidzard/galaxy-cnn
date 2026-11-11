from google.colab import drive
drive.mount('/content/drive')

import shutil, os, zipfile

if os.path.exists('/content/dataset'):
    shutil.rmtree('/content/dataset')

with zipfile.ZipFile('/content/drive/MyDrive/images_gz2.zip', 'r') as z:
    z.extractall('/content/dataset')
print("Gotovo!")

for item in os.listdir('/content/dataset/images_gz2'):
    path = f'/content/dataset/images_gz2/{item}'
    count = len(os.listdir(path))
    print(f"  {item}/ — {count} fajlova")

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

os.makedirs('grafici', exist_ok=True)

main_path  = '/content/dataset/images_gz2'
img_size   = (96, 96)    # brze od 150x150
batch_size = 128

train_ds, val_ds = image_dataset_from_directory(
    main_path, subset='both', validation_split=0.2,
    image_size=img_size, batch_size=batch_size, shuffle=True, seed=123
)
classes     = train_ds.class_names
num_classes = len(classes)
print(f'Classes: {classes}')

# Cache + prefetch — prva epoha ucitava u RAM, ostale idu iz RAM-a
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# ── Fast count from the folder structure (without reading the images) ────
print("\n--- Class distribution ---")
all_counts   = {}
train_counts = {}
val_counts   = {}

for i, cls in enumerate(classes):
    cls_path = os.path.join(main_path, cls)
    n        = len(glob.glob(os.path.join(cls_path, '*.jpg')) +
                   glob.glob(os.path.join(cls_path, '*.png')) +
                   glob.glob(os.path.join(cls_path, '*.jpeg')))
    train_n  = int(n * 0.8)
    val_n    = n - train_n
    all_counts[i]   = n
    train_counts[i] = train_n
    val_counts[i]   = val_n

total = sum(all_counts.values())
for i, cls in enumerate(classes):
    n = all_counts[i]
    print(f"  {cls}: {n} images ({100*n/total:.1f}%)")
print(f"Omjer max/min: {max(all_counts.values())/min(all_counts.values()):.2f}")

# ── Class weight iz poznatih brojeva ─────────────────────────
all_labels = np.array([i for i in range(num_classes)
                        for _ in range(train_counts[i])])
cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=all_labels)
class_weight = {i: cw[i] for i in range(num_classes)}
print(f"Class weights: {class_weight}")


import shutil, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Drive and folders ───────────────────────────────────────────
drive_output = '/content/drive/MyDrive/galaxy_rezultati'
os.makedirs('grafici', exist_ok=True)
os.makedirs(drive_output, exist_ok=True)
drive_grafici = os.path.join(drive_output, 'grafici')
os.makedirs(drive_grafici, exist_ok=True)

def sacuvaj(filename):
    """Save the figure locally and to Drive."""
    local = f'grafici/{filename}'
    plt.savefig(local, dpi=150, bbox_inches='tight')
    shutil.copy(local, os.path.join(drive_grafici, filename))
    print(f"  -> Sacuvano: {filename}")

# ── Histogram ────────────────────────────────────────────────
x = np.arange(num_classes)
w = 0.35
counts_train = [train_counts[i] for i in range(num_classes)]
counts_val   = [val_counts[i]   for i in range(num_classes)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(classes, [all_counts[i] for i in range(num_classes)],
            color=['#4C72B0','#DD8452','#55A868'])
axes[0].set_title('Samples per class (whole dataset)')
axes[0].set_ylabel('Image count')
for i, v in enumerate([all_counts[i] for i in range(num_classes)]):
    axes[0].text(i, v + total*0.003, str(v), ha='center', fontweight='bold')
axes[1].bar(x - w/2, counts_train, w, label='Train', color='#4C72B0')
axes[1].bar(x + w/2, counts_val,   w, label='Val',   color='#DD8452')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].set_title('Train / Validation split')
axes[1].set_ylabel('Image count')
axes[1].legend()
plt.tight_layout()
sacuvaj('01_class_histogram.png')
plt.show()

# ── One example from each class ─────────────────────────
samples = {}
for imgs, labels in train_ds:
    for img, lab in zip(imgs, labels):
        l = int(lab.numpy())
        if l not in samples:
            samples[l] = img.numpy().astype('uint8')
        if len(samples) == num_classes:
            break
    if len(samples) == num_classes:
        break

fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
fig.suptitle('One example from each class', fontsize=14)
for i in range(num_classes):
    axes[i].imshow(samples[i])
    axes[i].set_title(classes[i], fontsize=12)
    axes[i].axis('off')
plt.tight_layout()
sacuvaj('02_class_examples.png')
plt.show()

# ── Augmentacija ──────────────────────────────────────────────
data_augmentation = Sequential([
    layers.Input((img_size[0], img_size[1], 3)),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.1),
], name='augmentacija')

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Augmented examples (class: ' + classes[0] + ')', fontsize=13)
for ax in axes.flat:
    aug = data_augmentation(
        tf.expand_dims(tf.cast(samples[0], tf.float32), 0)
    )[0].numpy().astype('uint8')
    ax.imshow(aug)
    ax.axis('off')
plt.tight_layout()
sacuvaj('03_augmentation.png')
plt.show()

# ── Model ─────────────────────────────────────────────────────
def build_model(num_classes):
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='GalaxyCNN')

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

model = build_model(num_classes)
model.summary()

# ── Treniranje ────────────────────────────────────────────────
es = EarlyStopping(monitor='val_accuracy', patience=10,
                   restore_best_weights=True, verbose=1)

print("\nPocetak treniranja...")
history = model.fit(
    train_ds, epochs=50, validation_data=val_ds,
    callbacks=[es], class_weight=class_weight, verbose=1
)

# ── Cuvanje modela nakon treniranja ──────────────────────
model.save('galaxy_cnn_model.keras')
shutil.copy('galaxy_cnn_model.keras', os.path.join(drive_output, 'galaxy_cnn_model.keras'))
print("  -> Model sacuvan na Drive ODMAH nakon treniranja!")

# ── Performance plots ────────────────────────────────────────
ep = range(1, len(history.history['accuracy']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Performanse tokom treniranja', fontsize=14)
axes[0].plot(ep, history.history['accuracy'],     label='Train', marker='o', markersize=3)
axes[0].plot(ep, history.history['val_accuracy'], label='Val',   marker='o', markersize=3)
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoha')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, history.history['loss'],     label='Train', marker='o', markersize=3)
axes[1].plot(ep, history.history['val_loss'], label='Val',   marker='o', markersize=3)
axes[1].set_title('Gubitak (Loss)')
axes[1].set_xlabel('Epoha')
axes[1].set_ylabel('Gubitak')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
sacuvaj('04_training_curves.png')
plt.show()

# ── Predikcije ────────────────────────────────────────────────
def get_predictions(ds):
    y_true, y_pred, imgs_list = [], [], []
    for imgs, labels in ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        imgs_list.append(imgs.numpy())
    return (np.array(y_true), np.array(y_pred),
            np.concatenate(imgs_list, axis=0).astype('uint8'))

print("\nPredikcije na trening skupu...")
y_train_true, y_train_pred, imgs_train = get_predictions(train_ds)
print("Predikcije na validacionom skupu...")
y_val_true, y_val_pred, imgs_val = get_predictions(val_ds)

# ── Metrike ───────────────────────────────────────────────────
def print_and_return_metrics(y_true, y_pred, naziv):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'='*55}\n  METRIKE — {naziv}\n{'='*55}")
    print(f"  Accuracy     (Accuracy) : {acc*100:.2f}%")
    print(f"  Preciznost  (Precision): {prec*100:.2f}%")
    print(f"  Osetljivost (Recall)   : {rec*100:.2f}%")
    print(f"  F1-skor                : {f1*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    return acc, prec, rec, f1

acc_t, prec_t, rec_t, f1_t = print_and_return_metrics(y_train_true, y_train_pred, "TRENING SKUP")
acc_v, prec_v, rec_v, f1_v = print_and_return_metrics(y_val_true,   y_val_pred,   "VALIDACIONI SKUP")

# ── Cuvanje metrika ─────────────────────────────────────
with open(os.path.join(drive_output, 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("METRIKE KLASIFIKACIJE\n" + "="*55 + "\n")
    f.write(f"{'Metrika':<25} {'Train':>10} {'Val':>10}\n" + "-"*45 + "\n")
    f.write(f"{'Accuracy':<25} {acc_t*100:>9.2f}% {acc_v*100:>9.2f}%\n")
    f.write(f"{'Preciznost (macro)':<25} {prec_t*100:>9.2f}% {prec_v*100:>9.2f}%\n")
    f.write(f"{'Osetljivost (macro)':<25} {rec_t*100:>9.2f}% {rec_v*100:>9.2f}%\n")
    f.write(f"{'F1-skor (macro)':<25} {f1_t*100:>9.2f}% {f1_v*100:>9.2f}%\n")
    f.write("\nTRENING po klasama:\n")
    f.write(classification_report(y_train_true, y_train_pred, target_names=classes, zero_division=0))
    f.write("\nVALIDACIONI po klasama:\n")
    f.write(classification_report(y_val_true, y_val_pred, target_names=classes, zero_division=0))
print("  -> Metrike sacuvane na Drive!")

# ── Matrice konfuzije ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Matrice konfuzije', fontsize=14)
for ax, y_t, y_p, title in [
    (axes[0], y_train_true, y_train_pred, 'Trening skup (normalizovana)'),
    (axes[1], y_val_true,   y_val_pred,   'Validacioni skup (normalizovana)')
]:
    cm = confusion_matrix(y_t, y_p, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(title)
plt.tight_layout()
sacuvaj('05_confusion_matrices.png')
plt.show()

# ── Correct and incorrect classifications ──────────────────────────
def show_examples(imgs, y_true, y_pred, title, filename, n=5):
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx   = np.where(y_true != y_pred)[0]
    rng = np.random.default_rng(42)
    sel_c = rng.choice(correct_idx, min(n, len(correct_idx)), replace=False)
    sel_w = rng.choice(wrong_idx,   min(n, len(wrong_idx)),   replace=False)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
    fig.suptitle(title, fontsize=13)
    for col, idx in enumerate(sel_c):
        ax = axes[0, col]
        ax.imshow(imgs[idx])
        ax.set_title(f"T: {classes[int(y_true[idx])]}\nP: {classes[int(y_pred[idx])]}", fontsize=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('green'); spine.set_linewidth(3)
    for col, idx in enumerate(sel_w):
        ax = axes[1, col]
        ax.imshow(imgs[idx])
        ax.set_title(f"T: {classes[int(y_true[idx])]}\nP: {classes[int(y_pred[idx])]}", fontsize=8, color='red')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('red'); spine.set_linewidth(3)
    correct_patch = mpatches.Patch(color='green', label='Dobro klasifikovano')
    wrong_patch   = mpatches.Patch(color='red',   label='Lose klasifikovano')
    fig.legend(handles=[correct_patch, wrong_patch], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    sacuvaj(filename)
    plt.show()

show_examples(imgs_val, y_val_true, y_val_pred,
              'Correct (green) and incorrect (red) classifications',
              '06_classification_examples.png', n=5)

# ── Zavrsni ispis ─────────────────────────────────────────────
print(f"\nSve sacuvano na Drive: {drive_output}")
print("="*55)
print(f"  Accuracy     : Train {acc_t*100:.2f}%  |  Val {acc_v*100:.2f}%")
print(f"  Preciznost  : Train {prec_t*100:.2f}%  |  Val {prec_v*100:.2f}%")
print(f"  Osetljivost : Train {rec_t*100:.2f}%  |  Val {rec_v*100:.2f}%")
print(f"  F1-skor     : Train {f1_t*100:.2f}%  |  Val {f1_v*100:.2f}%")
print("="*55)


# ── 0. Mount Drive ────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 1. Unzip ako treba ───────────────────────────────────────
import zipfile, os
if not os.path.exists('/content/dataset/images_gz2'):
    with zipfile.ZipFile('/content/drive/MyDrive/images_gz2.zip', 'r') as z:
        z.extractall('/content/dataset')
    print("Unzipovano!")
else:
    print("Dataset vec postoji.")

# ── 2. Ucitaj dataset ────────────────────────────────────────
import tensorflow as tf

data_dir = '/content/dataset/images_gz2'
img_size = (96, 96)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

classes = train_ds.class_names
num_classes = len(classes)
print(f"Classes: {classes}, Class count: {num_classes}")

# ── 3. Ucitaj sacuvani model ─────────────────────────────────
import shutil, gc
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

drive_output = '/content/drive/MyDrive/galaxy_rezultati'
os.makedirs('grafici', exist_ok=True)
os.makedirs(drive_output, exist_ok=True)
drive_grafici = os.path.join(drive_output, 'grafici')
os.makedirs(drive_grafici, exist_ok=True)

def sacuvaj(filename):
    local = f'grafici/{filename}'
    plt.savefig(local, dpi=150, bbox_inches='tight')
    shutil.copy(local, os.path.join(drive_grafici, filename))
    print(f"  -> Sacuvano: {filename}")

print("Ucitavam model sa Drive...")
model = load_model(os.path.join(drive_output, 'galaxy_cnn_model.keras'))
print("Model ucitan!")

# ── 4. Predikcije  ──────────────────────────────
def get_predictions_light(ds):
    y_true, y_pred = [], []
    for imgs, labels in ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)

print("\nPredikcije na trening skupu...")
y_train_true, y_train_pred = get_predictions_light(train_ds)
gc.collect()
print("Predikcije na validacionom skupu...")
y_val_true, y_val_pred = get_predictions_light(val_ds)
gc.collect()

# ── 6. Metrike ────────────────────────────────────────────────
def print_and_return_metrics(y_true, y_pred, naziv):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'='*55}\n  METRIKE — {naziv}\n{'='*55}")
    print(f"  Accuracy     (Accuracy) : {acc*100:.2f}%")
    print(f"  Preciznost  (Precision): {prec*100:.2f}%")
    print(f"  Osetljivost (Recall)   : {rec*100:.2f}%")
    print(f"  F1-skor                : {f1*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    return acc, prec, rec, f1

acc_t, prec_t, rec_t, f1_t = print_and_return_metrics(y_train_true, y_train_pred, "TRENING SKUP")
acc_v, prec_v, rec_v, f1_v = print_and_return_metrics(y_val_true,   y_val_pred,   "VALIDACIONI SKUP")

# ── 7. Sacuvaj metrike ───────────────────────────────────────
with open(os.path.join(drive_output, 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("METRIKE KLASIFIKACIJE\n" + "="*55 + "\n")
    f.write(f"{'Metrika':<25} {'Train':>10} {'Val':>10}\n" + "-"*45 + "\n")
    f.write(f"{'Accuracy':<25} {acc_t*100:>9.2f}% {acc_v*100:>9.2f}%\n")
    f.write(f"{'Preciznost (macro)':<25} {prec_t*100:>9.2f}% {prec_v*100:>9.2f}%\n")
    f.write(f"{'Osetljivost (macro)':<25} {rec_t*100:>9.2f}% {rec_v*100:>9.2f}%\n")
    f.write(f"{'F1-skor (macro)':<25} {f1_t*100:>9.2f}% {f1_v*100:>9.2f}%\n")
    f.write("\nTRENING po klasama:\n")
    f.write(classification_report(y_train_true, y_train_pred, target_names=classes, zero_division=0))
    f.write("\nVALIDACIONI po klasama:\n")
    f.write(classification_report(y_val_true, y_val_pred, target_names=classes, zero_division=0))
print("  -> Metrike sacuvane na Drive!")

# ── 8. Matrice konfuzije ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Matrice konfuzije', fontsize=14)
for ax, y_t, y_p, title in [
    (axes[0], y_train_true, y_train_pred, 'Trening skup (normalizovana)'),
    (axes[1], y_val_true,   y_val_pred,   'Validacioni skup (normalizovana)')
]:
    cm = confusion_matrix(y_t, y_p, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(title)
plt.tight_layout()
sacuvaj('05_confusion_matrices.png')
plt.show()

# ── 9. Correct and incorrect classifications ─────────
def show_examples_light(ds, y_true, y_pred, title, filename, n=5):
    correct_idx = set(np.random.default_rng(42).choice(
        np.where(y_true == y_pred)[0], min(n, np.sum(y_true == y_pred)), replace=False))
    wrong_idx = set(np.random.default_rng(42).choice(
        np.where(y_true != y_pred)[0], min(n, np.sum(y_true != y_pred)), replace=False))
    target_indices = correct_idx | wrong_idx

    collected = {}
    idx = 0
    for imgs, labels in ds:
        for img in imgs:
            if idx in target_indices:
                collected[idx] = img.numpy().astype('uint8')
            idx += 1
            if len(collected) == len(target_indices):
                break
        if len(collected) == len(target_indices):
            break

    correct_list = sorted(correct_idx)
    wrong_list = sorted(wrong_idx)

    fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
    fig.suptitle(title, fontsize=13)
    for col, i in enumerate(correct_list[:n]):
        ax = axes[0, col]
        ax.imshow(collected[i])
        ax.set_title(f"T: {classes[int(y_true[i])]}\nP: {classes[int(y_pred[i])]}", fontsize=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('green'); spine.set_linewidth(3)
    for col, i in enumerate(wrong_list[:n]):
        ax = axes[1, col]
        ax.imshow(collected[i])
        ax.set_title(f"T: {classes[int(y_true[i])]}\nP: {classes[int(y_pred[i])]}", fontsize=8, color='red')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('red'); spine.set_linewidth(3)
    correct_patch = mpatches.Patch(color='green', label='Dobro klasifikovano')
    wrong_patch   = mpatches.Patch(color='red',   label='Lose klasifikovano')
    fig.legend(handles=[correct_patch, wrong_patch], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    sacuvaj(filename)
    plt.show()
    del collected
    gc.collect()

show_examples_light(val_ds, y_val_true, y_val_pred,
                    'Correct (green) and incorrect (red) classifications',
                    '06_classification_examples.png', n=5)

# ── 10. Zavrsni ispis ────────────────────────────────────────
print(f"\nSve sacuvano na Drive: {drive_output}")
print("="*55)
print(f"  Accuracy     : Train {acc_t*100:.2f}%  |  Val {acc_v*100:.2f}%")
print(f"  Preciznost  : Train {prec_t*100:.2f}%  |  Val {prec_v*100:.2f}%")
print(f"  Osetljivost : Train {rec_t*100:.2f}%  |  Val {rec_v*100:.2f}%")
print(f"  F1-skor     : Train {f1_t*100:.2f}%  |  Val {f1_v*100:.2f}%")
print("="*55)
