# Galaxy CNN — galaxy morphology classification

A convolutional neural network that sorts **Galaxy Zoo 2** images into three
morphological classes: **Barred** (barred spiral), **Elliptical**, and **Spiral**
(spiral without a bar).

Coursework for Neural Networks at the School of Electrical Engineering, University
of Belgrade.

## Authors

- **Aleksa Mikić** ([@AleksaMikic](https://github.com/AleksaMikic))
- **David M. Marković** ([@Dawidzard](https://github.com/Dawidzard))

The project was built jointly, before the repository was published on GitHub.

Student numbers: `203/2023`, `134/2023`.

---

## Results

Trained on **239,029 images** (80% train / 20% validation).

| Metric | Train | Validation |
|---|---|---|
| Accuracy | 75.14% | **75.10%** |
| Precision (macro) | 73.90% | 73.97% |
| Recall (macro) | 73.99% | 73.91% |
| F1 (macro) | 73.62% | 73.62% |

Per class, on validation:

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Barred | 0.68 | 0.69 | 0.69 | 9,166 |
| Elliptical | 0.77 | 0.88 | 0.82 | 19,578 |
| Spiral | 0.77 | 0.65 | 0.70 | 19,061 |

Train and validation are practically identical (75.14% against 75.10%), so the model
is **not overfitting** — what limits it is capacity and resolution, not memorisation.
The weakest class is `Barred`: the bar is a fine central structure that is often lost
at 96×96, and it is also the least represented class in the dataset (19% of samples).

Figures and the full `classification_report` are in [`results/`](results/):

| File | Contents |
|---|---|
| `01_class_histogram.png` | Class distribution and the train/validation split |
| `02_class_examples.png` | One example from each class |
| `03_augmentation.png` | Augmented examples |
| `04_training_curves.png` | Accuracy and loss per epoch |
| `05_confusion_matrices.png` | Confusion matrices, normalised |
| `06_classification_examples.png` | Correct (green) and incorrect (red) classifications |
| `metrics.txt` | Metrics and the per-class report |

Full write-up: [`docs/report-sr.pdf`](docs/report-sr.pdf) — in Serbian.

---

## Architecture

`GalaxyCNN` — seven convolutional layers in four blocks, with augmentation and
normalisation built into the model itself:

```
Augmentation (RandomFlip h+v, RandomRotation 0.3, RandomZoom 0.1)
Rescaling(1/255)

Conv2D(32)  + BN  ->  Conv2D(32)  + BN  ->  MaxPool
Conv2D(64)  + BN  ->  Conv2D(64)  + BN  ->  MaxPool
Conv2D(128) + BN  ->  Conv2D(128) + BN  ->  MaxPool
Conv2D(256) + BN  ->  MaxPool

Dropout(0.4) -> Flatten -> Dense(256, relu) -> Dropout(0.4) -> Dense(3, softmax)
```

Every `Conv2D` is 3×3, `padding='same'`, ReLU.

| Hyperparameter | Value |
|---|---|
| Image size | 96 × 96 |
| Batch size | 128 |
| Optimiser | Adam (default lr = 0.001) |
| Loss | SparseCategoricalCrossentropy |
| Epochs | 50, with EarlyStopping (`patience=10` on `val_accuracy`) |
| Class weights | `balanced` — the dataset is not |
| Split seed | 123 |

Because augmentation and `Rescaling` are **layers of the model**, images are passed
in at inference in the `[0, 255]` range — do not divide them by 255 yourself.

---

## Repository layout

```
.
├── train.py              # Train the model from scratch
├── evaluate.py           # Metrics, confusion matrices, example classifications
├── predict.py            # Classify one image or a whole folder
├── prepare_dataset.py    # Sort GZ2 images into one folder per class
├── models/
│   └── galaxy_cnn_model.keras   # The trained model (35 MB)
├── results/              # Figures and metrics
├── docs/
│   └── report-sr.pdf     # Project report, in Serbian
├── data/
│   ├── README.md         # How to obtain the dataset, and how to cite it
│   └── GALAXY_ZOO_README.txt    # The Galaxy Zoo team's own README
└── legacy/               # The original scripts as they were run, for reference
```

---

## Running it

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 2. Classify an image with the trained model

No dataset needed — the model is in the repository:

```bash
python predict.py my_galaxy.jpg
```

```
my_galaxy.jpg
  Prediction: Spiral (82.4% confidence)
  Breakdown:  Barred:  11.3%  Elliptical:   6.3%  Spiral:  82.4%
```

### 3. Prepare the dataset, to train

The dataset is not in the repository (243,434 images, about 5 GB). Downloading and
citing it are covered in [`data/README.md`](data/README.md). Once `images_gz2.zip`
is unpacked and the class CSV downloaded:

```bash
python prepare_dataset.py --images-dir images --csv gz2_classes.csv --out-dir images_gz2
```

which gives:

```
images_gz2/
    Barred/      ~45,600 images
    Elliptical/  ~97,600 images
    Spiral/      ~95,800 images
```

### 4. Train

```bash
python train.py --data-dir images_gz2
```

Training on the full dataset was done on a Google Colab GPU. On a CPU it takes
unreasonably long — to try it out, shrink the dataset (`--limit-per-class` in
`prepare_dataset.py`) or cut the epochs.

### 5. Evaluate

```bash
python evaluate.py --data-dir images_gz2 --model models/galaxy_cnn_model.keras
```

> **Important:** `--seed` and `--img-size` must match the values used for training
> (123 and 96). `validation_split` divides the dataset deterministically from the
> seed, so a different seed gives a different split — and then the "validation" set
> contains images the model was trained on, and the metrics look better than they are.

---

## Dataset and citation

The images and labels come from **Galaxy Zoo 2** and are not our work. If you use the
dataset, cite the papers listed in [`data/README.md`](data/README.md) — Willett et al.
(2013) and Hart et al. (2016).

## Notes

- `models/galaxy_cnn_model.keras` is 35 MB and is committed as an ordinary file (under
  GitHub's 100 MB limit), without Git LFS.
- The figures in `results/` come from the original Colab run, not from re-running
  these scripts.
