# test.py
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

from models import ResnetGenerator


def denorm(x):
    return (x + 1) / 2.0  # [-1,1] -> [0,1]


def load_generator(ckpt_path, device, size=256):
    netG = ResnetGenerator().to(device)
    state = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    return netG


def make_transform(size):
    return T.Compose([
        T.Resize(size, Image.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def run_inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = Path(opt.ckpt)
    input_dir = Path(opt.input_dir)
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load G_A2B (photo -> painting)
    netG = load_generator(ckpt_path, device, size=opt.size)
    transform = make_transform(opt.size)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    img_paths.sort()

    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(img_paths)} images. Saving to {out_dir}")

    with torch.no_grad():
        for i, path in enumerate(img_paths, 1):
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
            fake = netG(x)
            fake = denorm(fake.clamp(-1, 1))  # back to [0,1]

            out_path = out_dir / path.name
            vutils.save_image(fake, out_path)
            if i % 10 == 0 or i == len(img_paths):
                print(f"[{i}/{len(img_paths)}] saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to G_A2B_XX.pth checkpoint")
    p.add_argument("--input_dir", type=str, required=True,
                   help="folder with input photos")
    p.add_argument("--out_dir", type=str, required=True,
                   help="folder to save painted outputs")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    run_inference(args)
