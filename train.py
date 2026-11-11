"""
Train the CNN for galaxy morphology classification (Galaxy Zoo 2).

Classes: Barred / Elliptical / Spiral

Expected layout:

    images_gz2/
        Barred/      *.jpg
        Elliptical/  *.jpg
        Spiral/      *.jpg

Run:
    python train.py --data-dir images_gz2

The architecture and hyperparameters here are the ones that produced
models/galaxy_cnn_model.keras; its figures are in results/.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='images_gz2',
                   help='Folder sa podfolderima po klasama (default: images_gz2)')
    p.add_argument('--out-dir', default='results',
                   help='Gde se cuvaju grafici (default: results)')
    p.add_argument('--model-out', default='models/galaxy_cnn_model.keras',
                   help='Path to the saved model')
    p.add_argument('--img-size', type=int, default=96,
                   help='Image size, NxN (default: 96)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10,
                   help='EarlyStopping patience na val_accuracy')
    p.add_argument('--seed', type=int, default=123,
                   help='Seed za train/val split. VAZNO: evaluate.py mora '
                        'koristiti isti seed da val skup ostane neviden.')
    return p.parse_args()


def count_images_per_class(data_dir, classes, val_split=0.2):
    """Fast count from the folder structure, without reading the images."""
    all_counts, train_counts, val_counts = {}, {}, {}
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        n = len(glob.glob(os.path.join(cls_path, '*.jpg')) +
                glob.glob(os.path.join(cls_path, '*.png')) +
                glob.glob(os.path.join(cls_path, '*.jpeg')))
        train_n = int(n * (1 - val_split))
        all_counts[i] = n
        train_counts[i] = train_n
        val_counts[i] = n - train_n
    return all_counts, train_counts, val_counts


def build_augmentation(img_size):
    return Sequential([
        layers.Input((img_size, img_size, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.1),
    ], name='augmentacija')


def build_model(num_classes, data_augmentation):
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),

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


def plot_class_histogram(classes, all_counts, train_counts, val_counts, out_path):
    num_classes = len(classes)
    total = sum(all_counts.values())
    x = np.arange(num_classes)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(classes, [all_counts[i] for i in range(num_classes)],
                color=['#4C72B0', '#DD8452', '#55A868'])
    axes[0].set_title('Samples per class (whole dataset)')
    axes[0].set_ylabel('Image count')
    for i in range(num_classes):
        v = all_counts[i]
        axes[0].text(i, v + total * 0.003, str(v), ha='center', fontweight='bold')

    axes[1].bar(x - w / 2, [train_counts[i] for i in range(num_classes)], w,
                label='Train', color='#4C72B0')
    axes[1].bar(x + w / 2, [val_counts[i] for i in range(num_classes)], w,
                label='Val', color='#DD8452')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_title('Train / Validation split')
    axes[1].set_ylabel('Image count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")


def plot_class_samples(train_ds, classes, out_path):
    """One example from each class."""
    num_classes = len(classes)
    samples = {}
    for imgs, labels in train_ds:
        for img, lab in zip(imgs, labels):
            samples.setdefault(int(lab.numpy()), img.numpy().astype('uint8'))
        if len(samples) == num_classes:
            break

    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
    fig.suptitle('One example from each class', fontsize=14)
    for i in range(num_classes):
        axes[i].imshow(samples[i])
        axes[i].set_title(classes[i], fontsize=12)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")
    return samples


def plot_augmentation(data_augmentation, sample, class_name, out_path):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(f'Augmented examples (class: {class_name})', fontsize=13)
    for ax in axes.flat:
        aug = data_augmentation(
            tf.expand_dims(tf.cast(sample, tf.float32), 0)
        )[0].numpy().astype('uint8')
        ax.imshow(aug)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")


def plot_history(history, out_path):
    ep = range(1, len(history.history['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Performanse tokom treniranja', fontsize=14)

    axes[0].plot(ep, history.history['accuracy'], label='Train', marker='o', markersize=3)
    axes[0].plot(ep, history.history['val_accuracy'], label='Val', marker='o', markersize=3)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoha')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history.history['loss'], label='Train', marker='o', markersize=3)
    axes[1].plot(ep, history.history['val_loss'], label='Val', marker='o', markersize=3)
    axes[1].set_title('Gubitak (Loss)')
    axes[1].set_xlabel('Epoha')
    axes[1].set_ylabel('Gubitak')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")


def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)

    # Load the data
    train_ds, val_ds = image_dataset_from_directory(
        args.data_dir, subset='both', validation_split=0.2,
        image_size=img_size, batch_size=args.batch_size,
        shuffle=True, seed=args.seed
    )
    classes = train_ds.class_names
    num_classes = len(classes)
    print(f'Classes: {classes}')

    # Prefetch: the first epoch reads from disk, later ones come from the pipeline buffer
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Class distribution
    print("\n--- Class distribution ---")
    all_counts, train_counts, val_counts = count_images_per_class(args.data_dir, classes)
    total = sum(all_counts.values())
    for i, cls in enumerate(classes):
        print(f"  {cls}: {all_counts[i]} images ({100 * all_counts[i] / total:.1f}%)")
    print(f"Omjer max/min: {max(all_counts.values()) / min(all_counts.values()):.2f}")

    # Class weights (dataset je neizbalansiran)
    all_labels = np.array([i for i in range(num_classes)
                           for _ in range(train_counts[i])])
    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=all_labels)
    class_weight = {i: cw[i] for i in range(num_classes)}
    print(f"Class weights: {class_weight}")

    # Grafici pre treniranja
    plot_class_histogram(classes, all_counts, train_counts, val_counts,
                         os.path.join(args.out_dir, '01_class_histogram.png'))
    samples = plot_class_samples(train_ds, classes,
                                 os.path.join(args.out_dir, '02_class_examples.png'))

    data_augmentation = build_augmentation(args.img_size)
    plot_augmentation(data_augmentation, samples[0], classes[0],
                      os.path.join(args.out_dir, '03_augmentation.png'))

    # Model
    model = build_model(num_classes, data_augmentation)
    model.summary()

    # Treniranje
    es = EarlyStopping(monitor='val_accuracy', patience=args.patience,
                       restore_best_weights=True, verbose=1)

    print("\nPocetak treniranja...")
    history = model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds,
        callbacks=[es], class_weight=class_weight, verbose=1
    )

    model.save(args.model_out)
    print(f"  -> Model sacuvan: {args.model_out}")

    plot_history(history, os.path.join(args.out_dir, '04_training_curves.png'))

    print("\nTreniranje zavrseno. Za metrike i matrice konfuzije pokreni:")
    print(f"    python evaluate.py --data-dir {args.data_dir} "
          f"--model {args.model_out} --seed {args.seed}")


if __name__ == '__main__':
    main()
