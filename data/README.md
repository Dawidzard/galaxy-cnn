# Dataset

The images and labels come from the **Galaxy Zoo 2** project and are **not part of
this work**. The dataset is not in the repository — 243,434 images, around 5 GB.

## Downloading it

1. **Images** — `images_gz2.zip` (all GZ2 images, each file named after its
   `asset_id`):
   [Galaxy Zoo 2 images on Zenodo](https://zenodo.org/records/3565489)

2. **Labels** — the table of morphological classes, from
   [data.galaxyzoo.org](https://data.galaxyzoo.org):
   *"GZ2 — Table 1 — Normal-depth sample with new debiasing method"* (Hart et al.
   2016), plus `gz2_filename_mapping.csv` to join `objid` to `asset_id`.

   `prepare_dataset.py` needs a CSV with two columns:

   ```
   asset_id,gz2_class
   3,Sb
   4,Sc?l
   5,Er
   ```

3. **Sorting into classes** — from the repository root:

   ```bash
   python prepare_dataset.py --images-dir images --csv gz2_classes.csv --out-dir images_gz2
   ```

## Class mapping

GZ2 labels (`gz2_class`) are detailed morphology codes (`Er`, `Sc1t`, `SBc2m`, ...).
They are reduced to three classes by the first letters of the code:

| GZ2 code starts with | Class | Examples |
|---|---|---|
| `E` | Elliptical | `Er`, `Ei`, `Ec` |
| `SB` | Barred | `SBa`, `SBc2m` |
| `S` (anything else) | Spiral | `Sb`, `Sc1t`, `Sen` |
| `A` | *skipped* | artefacts and stars |

The resulting distribution:

| Class | Images | Share |
|---|---|---|
| Elliptical | ~97,600 | 41% |
| Spiral | ~95,800 | 40% |
| Barred | ~45,600 | 19% |
| **Total** | **~239,000** | |

The dataset is imbalanced, which is why `train.py` uses `class_weight='balanced'`.

The count is slightly below the CSV's row count because around 700 rows have no
corresponding image. The Galaxy Zoo team mentions this in their own README and
advises dropping those rows; `prepare_dataset.py` skips them and reports how many
at the end.

## Citation

If you use this dataset, you must cite:

- **Willett et al. (2013)**, *Galaxy Zoo 2: detailed morphological classifications for
  304,122 galaxies from the Sloan Digital Sky Survey*, MNRAS, 435, 2835.
  [DOI: 10.1093/mnras/stt1458](https://doi.org/10.1093/mnras/stt1458)
- **Hart et al. (2016)**, MNRAS, 461, 3663.
  [DOI: 10.1093/mnras/stw1588](https://doi.org/10.1093/mnras/stw1588)
  (if the debiased tables are used)
- The DOI of the image dataset itself.

The Galaxy Zoo team's own README, with the full terms and contact details, is in
[`GALAXY_ZOO_README.txt`](GALAXY_ZOO_README.txt).
