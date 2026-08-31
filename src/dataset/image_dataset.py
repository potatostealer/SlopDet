from pathlib import Path

from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

REAL_LABEL = 0
AIGEN_LABEL = 1


def list_images(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class BinaryImageDataset(Dataset):
    """(image path, label) pairs: real images are 0, AI generated images are 1."""

    def __init__(self, real_dir, aigen_dir):
        self.samples = (
            [(str(p), REAL_LABEL) for p in list_images(real_dir)]
            + [(str(p), AIGEN_LABEL) for p in list_images(aigen_dir)]
        )
        if not self.samples:
            raise ValueError(f"no images found in {real_dir} or {aigen_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
