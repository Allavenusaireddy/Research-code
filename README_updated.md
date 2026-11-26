# Hybrid Adversarial Attack Detection for X-IIoTID

This repository contains the Jupyter Notebook implementation of a hybrid adversarial-attack detection framework for a binary, ML-based Network Intrusion Detection System (NIDS). The supplied notebook is `full-code.ipynb`.

The pipeline trains a Deep Neural Network (DNN), Random Forest (RF), and calibrated Support Vector Machine (SVM), uses SHAP-based explanations and behavioural indicators, and trains a Gradient Boosting meta-classifier to distinguish clean traffic from four adversarial attack types.

## 1. Dataset

The notebook is configured for the **X-IIoTID dataset** and expects the CSV file:

```text
X-IIoTID dataset.csv
```

The configured target column is `class3`, with this binary mapping:

| Original label | Encoded label |
|---|---:|
| Attack | 0 |
| Normal | 1 |

The notebook drops rows with a missing target, removes exact duplicate rows, and removes the sibling label columns (`class1` and `class2`) when present. The remaining columns are used as input features.

**Dataset access:** obtain X-IIoTID from its official or authorised source. The dataset is not included in this repository. The notebook currently uses a Kaggle input path, so update that path if you run it in another environment.

## 2. Requirements

Recommended environment: Python 3.10 or 3.11, with a compatible PyTorch installation. The notebook does not pin package versions; use a mutually compatible set of the following dependencies:

- numpy
- pandas
- scikit-learn
- shap
- torch
- joblib
- jupyter (or JupyterLab)

Install the Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn shap torch joblib jupyter
```

For GPU acceleration, install the PyTorch build that matches your CUDA environment by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). CPU execution is also supported, but training and especially SHAP/adversarial evaluation may take substantially longer.

## 3. Dataset path configuration

In the notebook, find the dataset-loading statement:

```python
df = pd.read_csv(
    "/kaggle/input/datasets/venusaireddyalla/x-iiotid-dataset/X-IIoTID dataset.csv",
    low_memory=False
)
```

Change the path to the location of the CSV file in your environment. For example, for a local project:

```python
df = pd.read_csv("./data/X-IIoTID dataset.csv", low_memory=False)
```

In Kaggle, add the X-IIoTID dataset to the notebook using **Add Input**, then make sure the path matches the dataset directory shown in the Kaggle input panel.

The notebook is specifically configured for `label_col = "class3"` and the labels `Attack` and `Normal`. If your copy of the dataset uses different column names or label values, adjust the loading and label-mapping section before running the experiment.

## 4. Running the experiments

The notebook is organised as a staged workflow. Run the stages in order and retain the generated artifacts between stages.

### Stage A — Train the base and hybrid models

Run the notebook from the imports and dataset-loading section through the **Run 1 — Training Artifacts Saved** checkpoint.

This stage performs the following operations:

1. Loads the dataset, removes exact duplicate rows, and validates the binary target labels.
2. Splits the raw rows into base-training, base-validation, meta-training, meta-validation, and final-test groups before fitting preprocessing objects.
3. Fits preprocessing using the base-training split only:
   - numeric conversion and median imputation;
   - one-hot encoding for low-cardinality categorical features;
   - frequency encoding for high-cardinality categorical features;
   - MinMax scaling.
4. Trains the DNN and the RF and calibrated SVM ensemble models.
5. Builds SHAP explanations and baseline statistics using base-training data.
6. Generates adversarial examples for training and constructs the behavioural-signal datasets.
7. Trains the Gradient Boosting meta-classifier.
8. Saves the trained models, evaluation data, and experiment metadata.

The notebook's source split uses `random_state=42`. The final test split is 20% of the rows. A meta-data pool is reserved from the development portion (up to 5,000 rows, or approximately 10% of the development rows), and the remaining base pool is split into base-training and base-validation subsets. The meta pool is divided into meta-training and meta-validation subsets.

**Leakage-control note:** preprocessing is fitted only on base-training data. The notebook sets `source_col = None`, so the split is row-isolated rather than session-, device-, or source-group-isolated. Do not describe it as source-level isolation unless you configure a valid source identifier and implement the corresponding group split.

### Stage B — Run final-test evaluation in chunks

After Stage A completes, the notebook saves the artifacts to:

```text
/kaggle/working/hybrid_artifacts/
```

The evaluation stage reloads the trained models and saved test data, then evaluates the final-test rows in chunks to make the computation resumable.

The notebook defines:

```python
CHUNK_ID = 0
CHUNK_SIZE = 1000
RESULT_DIR = "/kaggle/working/hybrid_eval_chunks"
```

Run the evaluation stage once for each `CHUNK_ID` from `0` through `9`. Each run writes a file named `chunk_00.joblib` through `chunk_09.joblib` in `hybrid_eval_chunks`. If a chunk file already exists, the notebook stops rather than overwriting it; remove that specific file only when you intentionally want to recompute the chunk.

The evaluation code expects the saved artifacts to be available at `ARTIFACT_DIR`. In the supplied notebook, a later evaluation section sets this to a Kaggle input-dataset path:

```python
ARTIFACT_DIR = "/kaggle/input/datasets/venusaireddyalla/artifacts/hybrid_artifacts"
```

If you are evaluating in the same Kaggle session in which Stage A ran, change this to:

```python
ARTIFACT_DIR = "/kaggle/working/hybrid_artifacts"
```

Alternatively, publish or attach the saved `hybrid_artifacts` directory as a Kaggle dataset and use its actual input path. The artifact directory must contain at least `dnn_model.pt`, `evaluation_data.npz`, and `metadata.joblib`, as well as the saved classifier files used by the evaluation.

### Stage C — Combine chunks and calculate final metrics

After all ten chunks have completed, run the chunk-combination and final-metrics sections. The notebook checks that all ten chunk files are present and that their row ranges are contiguous before combining them.

The evaluation applies these four attack generators:

| Attack | Parameters used in the notebook |
|---|---|
| DeepSHAP-guided masked-PGD (reported in code as SHAP-PGD) | ε = 0.1, α = 0.02, 3 steps, top-k = 5 features |
| DeepFool | overshoot = 0.02, maximum 20 iterations |
| Low-perturbation PGD (Weak) | ε = 0.01, α = 0.005, 20 steps |
| Loss-guided Gaussian noise perturbation (Noise) | σ = 0.05, 50 candidate perturbations |

The hybrid detector uses 16 behavioural signals and a five-class meta-classifier output: clean/none, SHAP-PGD, DeepFool, Weak, and Noise.

The final evaluation code reports DNN binary classification metrics on the clean final-test set, exact four-way attack-type identification metrics, one-vs-rest metrics, ROC-AUC values, and a comparison with the rule-based detector. The evaluation stage is computationally expensive because it generates and analyses multiple adversarial variants for each test row.

## 5. Output files

### Training artifacts

The training stage writes the following files to `hybrid_artifacts/`:

| File | Description |
|---|---|
| `dnn_model.pt` | DNN weights and model dimensions |
| `rf_model.joblib` | Trained Random Forest |
| `svm_model.joblib` | Trained calibrated SVM |
| `meta_model.joblib` | Trained Gradient Boosting meta-classifier |
| `scaler.joblib` | Fitted MinMax scaler |
| `onehot_encoder.joblib` | Fitted one-hot encoder, when applicable |
| `evaluation_data.npz` | Final-test data and saved baseline/threshold arrays |
| `metadata.joblib` | Dataset, label mapping, split, attack, and signal metadata |

### Evaluation outputs

The chunked evaluation stage writes:

```text
hybrid_eval_chunks/
    chunk_00.joblib
    ...
    chunk_09.joblib
```

The combination and final reporting stage writes the combined results and metrics under the configured combined-results directory, including:

- `hybrid_predictions_10000.csv`
- `per_attack_metrics.csv`
- `attack_type_ovr_metrics.csv`
- `exact_attack_type_confusion_matrix.csv`
- `final_summary.joblib`

The notebook currently uses Kaggle-specific `/kaggle/working/` and `/kaggle/input/` paths. Change these paths when running locally or in another hosted notebook environment.

## 6. Reproducibility notes

- The notebook sets `random_state=42` for its train/test splits and the RF, calibrated SVM base estimator, and Gradient Boosting classifier.
- The notebook does not set all NumPy, PyTorch, CUDA, or data-loader random seeds globally. GPU operations and stochastic attack generation may therefore produce run-to-run differences. Exact bit-for-bit reproducibility is not guaranteed.
- Record the Python, PyTorch, CUDA, SHAP, NumPy, pandas, and scikit-learn versions used for a reported experiment.
- Keep the original dataset version and the generated `hybrid_artifacts` directory unchanged when reproducing the corresponding evaluation.
- Run all ten evaluation chunks exactly once, then combine them. Do not combine partial or mixed-run chunk files.
- Preserve the console logs and generated CSV/joblib outputs so the reported metrics can be checked.

