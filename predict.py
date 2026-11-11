"""
Classify one image, or every image in a folder, with a saved model.

Run:
    python predict.py image.jpg
    python predict.py my_folder/ --model models/galaxy_cnn_model.keras

Note: the model carries Rescaling(1./255) as a layer, so an image is NOT
divided by 255 before prediction -- it is passed in the [0, 255] range.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

# Class order is alphabetical, the same order image_dataset_from_directory
# assigns from the folder names during training.
CLASS_NAMES = ["Barred", "Elliptical", "Spiral"]

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', help='An image, or a folder of images')
    p.add_argument('--model', default='models/galaxy_cnn_model.keras')
    p.add_argument('--img-size', type=int, default=96,
                   help='Mora odgovarati treningu (default: 96)')
    return p.parse_args()


def collect_images(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(IMAGE_EXT))
    return [path]


def predict_one(model, img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=(img_size, img_size))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    return probs


def main():
    args = parse_args()

    images = collect_images(args.path)
    if not images:
        print(f"No images in: {args.path}")
        return

    print(f"Ucitavam model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    for img_path in images:
        probs = predict_one(model, img_path, args.img_size)
        pred_idx = int(np.argmax(probs))
        raspodela = "  ".join(f"{CLASS_NAMES[i]}: {probs[i] * 100:5.1f}%"
                              for i in range(len(CLASS_NAMES)))
        print(f"\n{os.path.basename(img_path)}")
        print(f"  Predikcija: {CLASS_NAMES[pred_idx]} "
              f"({probs[pred_idx] * 100:.1f}% sigurnosti)")
        print(f"  Raspodela:  {raspodela}")


if __name__ == '__main__':
    main()
