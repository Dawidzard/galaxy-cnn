"""
Evaluate a saved model: metrics, confusion matrices, example classifications.

Run:
    python evaluate.py --data-dir images_gz2 --model models/galaxy_cnn_model.keras

IMPORTANT: --seed and --img-size must match the values used for training.
validation_split divides the dataset deterministically from the seed, so a
different seed produces a different split -- and then the "validation" set
contains images the model was trained on, and every metric looks better
than it is.
"""

import argparse
import gc
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='images_gz2')
    p.add_argument('--model', default='models/galaxy_cnn_model.keras')
    p.add_argument('--out-dir', default='results')
    p.add_argument('--img-size', type=int, default=96,
                   help='Mora odgovarati treningu (default: 96)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=123,
                   help='Mora biti isti seed kao u train.py (default: 123)')
    p.add_argument('--examples', action='store_true',
                   help='Also save a plot of correct/incorrect classifications '
                        '(keeps the images in RAM; needs more memory)')
    return p.parse_args()


def load_datasets(data_dir, img_size, batch_size, seed):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset='both', validation_split=0.2,
        image_size=(img_size, img_size), batch_size=batch_size,
        shuffle=True, seed=seed
    )
    return train_ds, val_ds


def get_predictions(model, ds, keep_images=False):
    y_true, y_pred, imgs_list = [], [], []
    for imgs, labels in ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        if keep_images:
            imgs_list.append(imgs.numpy())
    images = (np.concatenate(imgs_list, axis=0).astype('uint8')
              if keep_images else None)
    return np.array(y_true), np.array(y_pred), images


def print_and_return_metrics(y_true, y_pred, classes, naziv):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'=' * 55}\n  METRIKE - {naziv}\n{'=' * 55}")
    print(f"  Accuracy     (Accuracy) : {acc * 100:.2f}%")
    print(f"  Preciznost  (Precision): {prec * 100:.2f}%")
    print(f"  Osetljivost (Recall)   : {rec * 100:.2f}%")
    print(f"  F1-skor                : {f1 * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    return acc, prec, rec, f1


def save_metrics(path, classes, train, val):
    (y_train_true, y_train_pred, acc_t, prec_t, rec_t, f1_t) = train
    (y_val_true, y_val_pred, acc_v, prec_v, rec_v, f1_v) = val

    with open(path, 'w', encoding='utf-8') as f:
        f.write("METRIKE KLASIFIKACIJE\n" + "=" * 55 + "\n")
        f.write(f"{'Metrika':<25} {'Train':>10} {'Val':>10}\n" + "-" * 45 + "\n")
        f.write(f"{'Accuracy':<25} {acc_t * 100:>9.2f}% {acc_v * 100:>9.2f}%\n")
        f.write(f"{'Preciznost (macro)':<25} {prec_t * 100:>9.2f}% {prec_v * 100:>9.2f}%\n")
        f.write(f"{'Osetljivost (macro)':<25} {rec_t * 100:>9.2f}% {rec_v * 100:>9.2f}%\n")
        f.write(f"{'F1-skor (macro)':<25} {f1_t * 100:>9.2f}% {f1_v * 100:>9.2f}%\n")
        f.write("\nTRENING po klasama:\n")
        f.write(classification_report(y_train_true, y_train_pred,
                                      target_names=classes, zero_division=0))
        f.write("\nVALIDACIONI po klasama:\n")
        f.write(classification_report(y_val_true, y_val_pred,
                                      target_names=classes, zero_division=0))
    print(f"  -> Sacuvano: {path}")


def plot_confusion_matrices(classes, train, val, out_path):
    y_train_true, y_train_pred = train
    y_val_true, y_val_pred = val

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Matrice konfuzije', fontsize=14)
    for ax, y_t, y_p, title in [
        (axes[0], y_train_true, y_train_pred, 'Trening skup (normalizovana)'),
        (axes[1], y_val_true, y_val_pred, 'Validacioni skup (normalizovana)')
    ]:
        cm = confusion_matrix(y_t, y_p, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")


def plot_examples(imgs, y_true, y_pred, classes, title, out_path, n=5):
    """Gornji red: dobro klasifikovane. Donji red: lose klasifikovane."""
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]
    rng = np.random.default_rng(42)
    sel_c = rng.choice(correct_idx, min(n, len(correct_idx)), replace=False)
    sel_w = rng.choice(wrong_idx, min(n, len(wrong_idx)), replace=False)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    fig.suptitle(title, fontsize=13)
    for row, (selection, color) in enumerate([(sel_c, 'green'), (sel_w, 'red')]):
        for col, idx in enumerate(selection):
            ax = axes[row, col]
            ax.imshow(imgs[idx])
            ax.set_title(f"T: {classes[int(y_true[idx])]}\nP: {classes[int(y_pred[idx])]}",
                         fontsize=8, color=None if row == 0 else 'red')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    fig.legend(handles=[mpatches.Patch(color='green', label='Dobro klasifikovano'),
                        mpatches.Patch(color='red', label='Lose klasifikovano')],
               loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Sacuvano: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds = load_datasets(args.data_dir, args.img_size,
                                     args.batch_size, args.seed)
    classes = train_ds.class_names
    print(f"Classes: {classes}")

    print(f"Ucitavam model: {args.model}")
    model = load_model(args.model)

    print("\nPredikcije na trening skupu...")
    y_train_true, y_train_pred, _ = get_predictions(model, train_ds)
    gc.collect()
    print("Predikcije na validacionom skupu...")
    y_val_true, y_val_pred, imgs_val = get_predictions(model, val_ds,
                                                       keep_images=args.examples)
    gc.collect()

    acc_t, prec_t, rec_t, f1_t = print_and_return_metrics(
        y_train_true, y_train_pred, classes, "TRENING SKUP")
    acc_v, prec_v, rec_v, f1_v = print_and_return_metrics(
        y_val_true, y_val_pred, classes, "VALIDACIONI SKUP")

    save_metrics(os.path.join(args.out_dir, 'metrics.txt'), classes,
                 (y_train_true, y_train_pred, acc_t, prec_t, rec_t, f1_t),
                 (y_val_true, y_val_pred, acc_v, prec_v, rec_v, f1_v))

    plot_confusion_matrices(classes,
                            (y_train_true, y_train_pred),
                            (y_val_true, y_val_pred),
                            os.path.join(args.out_dir, '05_confusion_matrices.png'))

    if args.examples:
        plot_examples(imgs_val, y_val_true, y_val_pred, classes,
                      'Correct (green) and incorrect (red) classifications',
                      os.path.join(args.out_dir, '06_classification_examples.png'))

    print("\n" + "=" * 55)
    print(f"  Accuracy     : Train {acc_t * 100:.2f}%  |  Val {acc_v * 100:.2f}%")
    print(f"  Preciznost  : Train {prec_t * 100:.2f}%  |  Val {prec_v * 100:.2f}%")
    print(f"  Osetljivost : Train {rec_t * 100:.2f}%  |  Val {rec_v * 100:.2f}%")
    print(f"  F1-skor     : Train {f1_t * 100:.2f}%  |  Val {f1_v * 100:.2f}%")
    print("=" * 55)


if __name__ == '__main__':
    main()
