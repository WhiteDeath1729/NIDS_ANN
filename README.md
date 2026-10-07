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
python src/predict.py --model models/demo_model.pt --file data/demo_data/malicious/malicious_000.bin --offset 0


For a quick smoke test, the defaults already point at the synthetic demo dataset:

bash
python scripts/make_demo_data.py
python src/train.py --epochs 30
python src/predict.py


## Important reproduction note

The paper's reported results came from its own network-traffic/file-content dataset and repeated 10-fold cross-validation. This repository does not contain that dataset, so results obtained on another dataset are not the paper's reported 98% accuracy. The paper reports average accuracy 0.98, precision 0.97, sensitivity 0.95 and average AUROC 0.98. It also reports a separate 400,000-file benign-data test with about 1.8% false positives.

Source: Shenfield et al., 2018.
