"""
Sanity check for the trained detector (CLAUDE.md section 10).
Run it AFTER training has finished and outputs/detector.pt exists.

What it does:
  1. Picks 3 real and 3 fake images from the validation set (fixed random seed).
  2. Prints the detector's probability "p_real" for each (near 1 = REAL, near 0 = AI-GENERATED).
  3. Checks the label direction: real images should get a HIGHER p_real than fake ones.
  4. Checks that no file name appears in both train/ and valid/ (no leaked images).

Usage:
    python sanity_check.py                                   # default data folder + outputs/detector.pt
    python sanity_check.py data/real_vs_fake/real-vs-fake
    python sanity_check.py data/dummy --weights outputs/smoke/detector.pt
"""
import argparse, os, random
import torch
from fgan_tonight import build_models, predict     # reuse the same model and image preprocessing

N_PER_CLASS = 3


def pick_images(folder, n):
    names = sorted(os.listdir(folder))
    random.shuffle(names)
    return [os.path.join(folder, f) for f in names[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data/real_vs_fake/real-vs-fake")
    ap.add_argument("--weights", default="outputs/detector.pt", help="trained detector file")
    args = ap.parse_args()

    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # load the trained detector (only D is needed, the generator is not used here)
    _, D = build_models(dev)
    D.load_state_dict(torch.load(args.weights, map_location=dev, weights_only=True))
    D.eval()

    # --- predictions on 3 real and 3 fake validation images ---
    scores = {}
    for cls in ("real", "fake"):
        scores[cls] = []
        print(f"\n--- {cls.upper()} validation images (true label: {cls}) ---")
        for path in pick_images(f"{args.data_dir}/valid/{cls}", N_PER_CLASS):
            r = predict(D, path, dev)
            scores[cls].append(r["p_real"])
            print(f"{os.path.basename(path):28s} p_real={r['p_real']:.3f}  ->  {r['label']}")

    # --- label direction check ---
    mean_real = sum(scores["real"]) / len(scores["real"])
    mean_fake = sum(scores["fake"]) / len(scores["fake"])
    print(f"\nmean p_real on real images: {mean_real:.3f}   on fake images: {mean_fake:.3f}")
    if mean_real > mean_fake:
        print("OK: real images score higher than fake ones (label direction looks right).")
    else:
        print("WARNING: real images do NOT score higher than fake ones. The labels may be flipped "
              "(expected fake=0, real=1) or the model has not learned anything.")
    print("Note: only 6 images. This is a quick direction check, not a measure of accuracy.")

    # --- train/valid overlap check (by file name) ---
    print("\n--- train/valid overlap check ---")
    for cls in ("real", "fake"):
        a = set(os.listdir(f"{args.data_dir}/train/{cls}"))
        b = set(os.listdir(f"{args.data_dir}/valid/{cls}"))
        print(f"{cls}: {len(a & b)} file names appear in both train and valid")


if __name__ == "__main__":
    main()
