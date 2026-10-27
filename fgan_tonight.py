"""
FGAN tonight-version: GAN-generated face detector.
Stage 1: short DCGAN run (adversarial training of Generator vs CNN Discriminator).
Stage 2: fine-tune the Discriminator as a real/fake detector on the Kaggle dataset.
Stage 3: evaluate clean + JPEG-compressed images (social media robustness).

Dataset: Kaggle "140k Real and Fake Faces" (xhlulu/140k-real-and-fake-faces)
Usage:
    python fgan_tonight.py data/real_vs_fake/real-vs-fake            # full run
    python fgan_tonight.py data/dummy --smoke                        # quick test
Folder layout expected: <DATA_DIR>/train/{real,fake}  <DATA_DIR>/valid/{real,fake}
"""
import argparse, io, os, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from torchvision.utils import save_image
from PIL import Image

# ---------- settings (change them here) ----------
IMG_SIZE = 64
NZ = 100             # size of the random noise vector fed to the generator
BATCH_SIZE = 128
N_TRAIN = 10000      # random subset of training images
N_VALID = 2000       # random subset of validation images
GAN_EPOCHS = 5
CLS_EPOCHS = 15
SMOKE_N = 200        # images used by --smoke
SMOKE_EPOCHS = 1     # epochs used by --smoke
NUM_WORKERS = 2      # background loader processes (safe on Windows because of the main() guard below)
SEED = 0


# ---------- data helpers (defined at top level so Windows workers can import them) ----------
class JPEG:
    """Simulates social-media compression: save as JPEG with random quality, then reload."""
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __call__(self, img):
        b = io.BytesIO()
        img.convert("RGB").save(b, "JPEG", quality=random.randint(self.lo, self.hi))
        b.seek(0)
        return Image.open(b).convert("RGB")


def make_transforms():
    norm = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(),
                      T.Normalize((.5,) * 3, (.5,) * 3)])        # pixels -> [-1, 1]
    train_tf = T.Compose([T.RandomApply([JPEG(30, 95)], p=0.5), norm])
    jpeg_tf = T.Compose([JPEG(30, 30), norm])
    return train_tf, norm, jpeg_tf


def random_subset(ds, n):
    idx = list(range(len(ds)))
    random.shuffle(idx)
    return Subset(ds, idx[:min(n, len(idx))])


# ---------- models ----------
def build_models(dev):
    # Generator: noise (100x1x1) -> 64x64 RGB image
    G = nn.Sequential(
        nn.ConvTranspose2d(NZ, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()).to(dev)

    def blk(i, o, bn=True):
        return [nn.Conv2d(i, o, 4, 2, 1, bias=not bn)] + ([nn.BatchNorm2d(o)] if bn else []) + [nn.LeakyReLU(.2, True)]

    # Discriminator = the CNN detector: 64x64 image -> one logit (high = real)
    D = nn.Sequential(*blk(3, 64, False), *blk(64, 128), *blk(128, 256), *blk(256, 512),
                      nn.Flatten(), nn.Dropout(.3), nn.Linear(512 * 4 * 4, 1)).to(dev)
    return G, D


@torch.no_grad()
def evaluate(D, ds, dev):
    """Returns accuracy, fake-precision, fake-recall and the confusion counts. 'fake' is the positive class."""
    D.eval()
    tp = fp = fn = tn = 0
    for x, y in DataLoader(ds, 256, num_workers=NUM_WORKERS):
        p = (D(x.to(dev)).view(-1) > 0).long().cpu()      # 1 = real, 0 = fake
        tp += ((p == 0) & (y == 0)).sum().item()
        fp += ((p == 0) & (y == 1)).sum().item()
        fn += ((p == 1) & (y == 0)).sum().item()
        tn += ((p == 1) & (y == 1)).sum().item()
    D.train()
    total = max(tp + fp + fn + tn, 1)
    return {"acc": (tp + tn) / total, "prec": tp / max(tp + fp, 1), "rec": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser(description="FGAN: GAN-generated face detector")
    ap.add_argument("data_dir", nargs="?", default="data/real_vs_fake/real-vs-fake",
                    help="folder that contains train/ and valid/ (each with real/ and fake/)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"quick test: ~{SMOKE_N} images, {SMOKE_EPOCHS} epoch, results go to outputs/smoke/")
    args = ap.parse_args()

    n_train, n_valid = (SMOKE_N, SMOKE_N) if args.smoke else (N_TRAIN, N_VALID)
    gan_epochs, cls_epochs = (SMOKE_EPOCHS, SMOKE_EPOCHS) if args.smoke else (GAN_EPOCHS, CLS_EPOCHS)
    out_dir = os.path.join("outputs", "smoke") if args.smoke else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "NO GPU (training will be slow)")
    if args.smoke:
        print(f"SMOKE TEST: {n_train} train / {n_valid} valid images, {gan_epochs} epoch(s)")

    # ---------- data ----------
    train_tf, clean_tf, jpeg_tf = make_transforms()
    # ImageFolder sorts classes alphabetically: fake=0, real=1
    train_ds = datasets.ImageFolder(f"{args.data_dir}/train", train_tf)
    print("classes:", train_ds.class_to_idx, "| train images:", len(train_ds))
    assert train_ds.class_to_idx == {"fake": 0, "real": 1}, "expected folders 'fake' and 'real'"
    train_sub = random_subset(train_ds, n_train)
    valid_clean = random_subset(datasets.ImageFolder(f"{args.data_dir}/valid", clean_tf), n_valid)
    valid_jpeg = Subset(datasets.ImageFolder(f"{args.data_dir}/valid", jpeg_tf), valid_clean.indices)

    # real-only loader for the GAN stage (random real images)
    real_idx = [i for i, t in enumerate(train_ds.targets) if t == 1]
    random.shuffle(real_idx)
    real_subset = Subset(train_ds, real_idx[:n_train])
    bs = min(BATCH_SIZE, len(real_subset))      # tiny datasets: never ask for a batch bigger than the data
    assert bs > 1, "need at least 2 real training images"
    real_loader = DataLoader(real_subset, bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    G, D = build_models(dev)
    bce = nn.BCEWithLogitsLoss()

    # ---------- stage 1: adversarial training ----------
    print("\n== Stage 1: DCGAN adversarial training ==")
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(.5, .999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(.5, .999))
    for ep in range(gan_epochs):
        t0 = time.time()
        for x, _ in real_loader:
            x = x.to(dev); n = x.size(0)
            z = torch.randn(n, NZ, 1, 1, device=dev)
            fake = G(z)
            # train D: real -> 1, generated -> 0
            oD.zero_grad()
            lD = bce(D(x).view(-1), torch.ones(n, device=dev)) + \
                 bce(D(fake.detach()).view(-1), torch.zeros(n, device=dev))
            lD.backward(); oD.step()
            # train G: make D say "real" for generated images
            oG.zero_grad()
            lG = bce(D(fake).view(-1), torch.ones(n, device=dev))
            lG.backward(); oG.step()
        print(f"epoch {ep+1}/{gan_epochs}  lossD={lD.item():.3f}  lossG={lG.item():.3f}  ({time.time()-t0:.0f}s)")

    G.eval()
    with torch.no_grad():
        save_image(G(torch.randn(64, NZ, 1, 1, device=dev)) * .5 + .5, f"{out_dir}/gan_samples.png", nrow=8)
    G.train()
    print(f"saved {out_dir}/gan_samples.png")

    # ---------- stage 2: fine-tune D as detector ----------
    print("\n== Stage 2: fine-tune discriminator as detector ==")
    cls_loader = DataLoader(train_sub, bs, shuffle=True, num_workers=NUM_WORKERS)
    oC = torch.optim.Adam(D.parameters(), 1e-4, betas=(.5, .999))
    for ep in range(cls_epochs):
        t0 = time.time()
        for x, y in cls_loader:
            x, y = x.to(dev), y.float().to(dev)
            oC.zero_grad()
            loss = bce(D(x).view(-1), y)             # label 1 = real
            loss.backward(); oC.step()
        r = evaluate(D, valid_clean, dev)
        print(f"epoch {ep+1}/{cls_epochs}  loss={loss.item():.3f}  valid acc={r['acc']:.3f}  ({time.time()-t0:.0f}s)")

    # ---------- stage 3: evaluation ----------
    print("\n== Stage 3: evaluation ==")
    rows = []
    for name, ds in [("clean", valid_clean), ("JPEG q=30", valid_jpeg)]:
        r = evaluate(D, ds, dev)
        rows.append((name, r))
        print(f"{name:10s} acc={r['acc']:.3f}  fake-precision={r['prec']:.3f}  fake-recall={r['rec']:.3f}")

    with open(f"{out_dir}/results.md", "w") as f:
        f.write("| Images | Accuracy | Precision (fake) | Recall (fake) |\n|---|---|---|---|\n")
        for name, r in rows:
            f.write(f"| {name} | {r['acc']:.3f} | {r['prec']:.3f} | {r['rec']:.3f} |\n")
        for name, r in rows:
            f.write(f"\nConfusion matrix, {name} (rows = true, columns = predicted):\n\n"
                    f"| | pred fake | pred real |\n|---|---|---|\n"
                    f"| true fake | {r['tp']} | {r['fn']} |\n| true real | {r['fp']} | {r['tn']} |\n")
    torch.save(D.state_dict(), f"{out_dir}/detector.pt")
    print(f"saved {out_dir}/results.md and {out_dir}/detector.pt")


# ---------- single-image inference (uses the model returned by build_models + loaded weights) ----------
def predict(D, path, dev="cpu"):
    D.eval()
    _, clean_tf, _ = make_transforms()
    x = clean_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        p_real = torch.sigmoid(D(x)).item()
    return {"p_real": p_real, "label": "REAL" if p_real > .5 else "AI-GENERATED"}


# On Windows, DataLoader workers re-import this file. This guard stops them from re-running training.
if __name__ == "__main__":
    main()
