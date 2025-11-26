# datasets.py
import random
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class UnpairedImageDataset(Dataset):
    """
    root/trainA/*.jpg (domain A)
    root/trainB/*.jpg (domain B)
    """
    def __init__(self, root: str, size: int = 256):
        super().__init__()
        root = Path(root)
        self.dir_A = root / "trainA"
        self.dir_B = root / "trainB"

        self.files_A: List[Path] = sorted(self.dir_A.glob("*.*"))
        self.files_B: List[Path] = sorted(self.dir_B.glob("*.*"))

        if len(self.files_A) == 0:
            raise RuntimeError(f"No images in {self.dir_A}")
        if len(self.files_B) == 0:
            raise RuntimeError(f"No images in {self.dir_B}")

        self.transform = T.Compose([
            T.Resize(int(size * 1.12), Image.BICUBIC),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB")

        return {
            "A": self.transform(img_A),
            "B": self.transform(img_B),
        }
