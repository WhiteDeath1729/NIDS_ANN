import argparse
from pathlib import Path
import torch
from ann_ids import make_model, bytes_to_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "demo_model.pt"
DEFAULT_FILE = PROJECT_ROOT / "data" / "demo_data" / "malicious" / "malicious_060.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--offset", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, map_location=device)
    offset = checkpoint["offset"] if args.offset is None else args.offset

    model = make_model(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x = torch.tensor(bytes_to_vector(args.file, offset)).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    label = "malicious" if p[1].item() >= 0.5 else "benign"

    print(f"Prediction : {label}")
    print(f"Benign     : {p[0].item():.4f}")
    print(f"Malicious  : {p[1].item():.4f}")

if __name__ == "__main__":
    main()
