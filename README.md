# ANN Intrusion Detection System — paper reproduction

A compact implementation of the architecture described in:
Shenfield, Day & Ayesh, "Intelligent intrusion detection systems using artificial neural networks", ICT Express 4 (2018), 95–99.

## Architecture
- Input: 1000 byte values
- Hidden layer 1: 30 neurons
- Hidden layer 2: 30 neurons
- Output: 2 classes (`benign`, `malicious`)
- Xavier/Glorot weight initialization
- Rprop optimizer
- Maximum 1000 epochs
- Learning-rate setting: 0.01 (kept as the requested/reference hyperparameter)
- Zero padding for files shorter than 1000 bytes

The paper explicitly says the exact file "magic numbers" at the start of files were avoided, but does not specify a universal byte offset. Therefore this implementation exposes `--offset` rather than silently inventing one.

## Project layout
shellcode_ann_ids/
├── README.md
├── requirements.txt
├── data/
│   └── demo_data/
│       ├── benign/
│       └── malicious/
├── models/
│   └── demo_model.pt
├── scripts/
│   └── make_demo_data.py
└── src/
    ├── ann_ids.py
    ├── train.py
    └── predict.py


data/
  my_dataset/
    benign/
      file1.bin
      ...
    malicious/
      shellcode1.bin
      ...


The code reads files as raw bytes, converts each selected byte to an integer in [0,255], takes 1000 contiguous bytes, and zero-pads shorter samples.

## Run

pip install -r requirements.txt

python scripts/make_demo_data.py
python src/train.py --data data/demo_data --epochs 1000 --offset 0 --save models/demo_model.pt
python src/predict.py --model models/demo_model.pt --data data/demo_data --batch-size 256 --output predictions.csv


For a quick smoke test, the defaults already point at the synthetic demo dataset:

bash
python scripts/make_demo_data.py
python src/train.py --epochs 30
python src/predict.py --data data/demo_data


### Batch prediction

`predict.py` can process an entire directory recursively instead of one file at a time.
Labels are inferred from any `benign` or `malicious` parent directory, so nested datasets are supported.
Inference is performed in batches and every prediction is written to a CSV containing the file, predicted class, probabilities, and (when available) the actual class.

```bash
# Evaluate a labelled dataset
python src/predict.py --data data/demo_data --batch-size 256 --output predictions.csv

# Predict arbitrary files without ground-truth labels
python src/predict.py --data path/to/unlabelled/files --output predictions.csv

# Predict several explicit files
python src/predict.py --file sample1.bin sample2.bin sample3.bin
```

Useful options:
- `--batch-size 256` controls how many files are passed through the ANN at once.
- `--threshold 0.5` controls the malicious probability threshold.
- `--no-print-files` suppresses per-file console output and prints only the summary.
- `--output predictions.csv` controls the CSV output location.

If the dataset contains both `benign` and `malicious` labels, the script also reports accuracy, precision, recall, AUROC, confusion matrix, and a classification report. If labels are absent, it performs prediction only.

## Important reproduction note

The paper's reported results came from its own network-traffic/file-content dataset and repeated 10-fold cross-validation. This repository does not contain that dataset, so results obtained on another dataset are not the paper's reported 98% accuracy. The paper reports average accuracy 0.98, precision 0.97, sensitivity 0.95 and average AUROC 0.98. It also reports a separate 400,000-file benign-data test with about 1.8% false positives.

Source: Shenfield et al., 2018.
