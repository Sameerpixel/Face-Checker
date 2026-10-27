"""
Makes a tiny DUMMY dataset of random-noise images, only to test that the pipeline runs.
It is NOT real data: results on it mean nothing.

Usage: python make_fake_dataset.py            (creates data/dummy/)
       python make_fake_dataset.py --n 100 --out data/dummy
"""
import argparse, os
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="images per class per split")
    ap.add_argument("--out", default="data/dummy")
    args = ap.parse_args()

    for split in ("train", "valid"):
        for cls in ("real", "fake"):
            folder = os.path.join(args.out, split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(args.n):
                # 64x64 RGB image made of random bytes = pure noise
                img = Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3))
                img.save(os.path.join(folder, f"{cls}_{i:04d}.jpg"), quality=95)
    print(f"created {args.n} images x 2 classes x 2 splits in {args.out}")


if __name__ == "__main__":
    main()
