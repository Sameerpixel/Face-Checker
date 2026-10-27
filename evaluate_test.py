"""
One-time evaluation of the trained detector on the held-out TEST split.

Why: the validation images were also used to choose the number of training epochs, so the
validation numbers may be slightly optimistic. The test/ folder was never used for training or
for any choice, so this gives the unbiased number to report. Run it ONCE, after training is final.
(If you change the model after seeing this result, the test split is no longer untouched.)

It uses ALL images in test/ (10,000 real + 10,000 fake in the Kaggle dataset), clean and with
JPEG quality 30, with the same model definition, preprocessing and metrics as fgan_tonight.py.

Usage:
    python evaluate_test.py
    python evaluate_test.py data/real_vs_fake/real-vs-fake --weights outputs/detector.pt
Writes outputs/test_results.md
"""
import argparse, os, random, time
import torch
from torchvision import datasets
from fgan_tonight import build_models, evaluate, make_transforms, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data/real_vs_fake/real-vs-fake")
    ap.add_argument("--weights", default="outputs/detector.pt")
    ap.add_argument("--split", default="test", help="folder to evaluate (default: test)")
    ap.add_argument("--out", default="outputs/test_results.md")
    args = ap.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "NO GPU (this will be slow)")

    _, D = build_models(dev)
    D.load_state_dict(torch.load(args.weights, map_location=dev, weights_only=True))
    D.eval()

    _, clean_tf, jpeg_tf = make_transforms()
    clean = datasets.ImageFolder(f"{args.data_dir}/{args.split}", clean_tf)
    jpeg = datasets.ImageFolder(f"{args.data_dir}/{args.split}", jpeg_tf)
    assert clean.class_to_idx == {"fake": 0, "real": 1}, "expected folders 'fake' and 'real'"
    n_fake, n_real = clean.targets.count(0), clean.targets.count(1)
    print(f"{args.split}/: {len(clean)} images ({n_fake} fake, {n_real} real)")

    rows = []
    for name, ds in [("clean", clean), ("JPEG q=30", jpeg)]:
        t0 = time.time()
        r = evaluate(D, ds, dev)
        rows.append((name, r))
        print(f"{name:10s} acc={r['acc']:.3f}  fake-precision={r['prec']:.3f}  fake-recall={r['rec']:.3f}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"Held-out {args.split} split: {len(clean)} images ({n_fake} fake, {n_real} real), weights: {args.weights}\n\n")
        f.write("| Images | Accuracy | Precision (fake) | Recall (fake) |\n|---|---|---|---|\n")
        for name, r in rows:
            f.write(f"| {name} | {r['acc']:.3f} | {r['prec']:.3f} | {r['rec']:.3f} |\n")
        for name, r in rows:
            f.write(f"\nConfusion matrix, {name} (rows = true, columns = predicted):\n\n"
                    f"| | pred fake | pred real |\n|---|---|---|\n"
                    f"| true fake | {r['tp']} | {r['fn']} |\n| true real | {r['fp']} | {r['tn']} |\n")
    print("saved", args.out)


if __name__ == "__main__":
    main()
