# legacy/

The original scripts, exactly as they were run while the project was being built.
They are kept for reference — to actually run anything, use `train.py`,
`evaluate.py` and `predict.py` in the repository root.

## `colab_main.py`

The final Google Colab script: the one that trained `models/galaxy_cnn_model.keras`
and produced the figures in `results/`. It is an export from a notebook, so it still
carries repeated imports, `drive.mount()` and `/content/...` paths.

It runs in two passes:

1. prepare the data, train, and save the model (`seed=123`, `batch_size=128`)
2. reload the saved model and evaluate it (`seed=42`, `batch_size=32`)

The second pass uses **a different seed from the first**, so `validation_split`
divides the dataset differently there and some of those "validation" images were
already seen during training. That is why `evaluate.py` defaults to `--seed 123` and
warns that the seed has to match the one used for training.

## `train_local_45k.py`

An earlier, simpler variant for training locally on a balanced subset — 45,000
images per class, 150×150, four convolutional blocks without the doubled Conv
layers, Adam at lr=1e-4, and no `Rescaling` layer. This is not the version that
produced the final model.
