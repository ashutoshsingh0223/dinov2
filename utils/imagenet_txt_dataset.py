"""
PyTorch Dataset for loading images and labels from a JSON file.

Expected JSON format:
    {
        "data_list": [
            {"image_path": "path/to/image1.jpg", "gt_label": 0},
            {"image_path": "path/to/image2.jpg", "gt_label": 1},
            ...
        ]
    }

Paths are relative to a given root directory.
"""

import json
import os
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetJsonDataset(Dataset):
    """Dataset that reads (image_path, label) pairs from a JSON file."""

    def __init__(
        self,
        json_file: str,
        root: str = "",
        image_key: str = "img_path",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Args:
            json_file: Path to JSON file with a "data_list" key.
            root: Root directory prepended to each image path.
            image_key: Key in each dict for the image path (default: "image_path").
            transform: Optional transform applied to the PIL image.
            target_transform: Optional transform applied to the label.
        """
        self.root = root
        self.image_key = image_key
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_json(json_file, image_key)

    @staticmethod
    def _load_json(json_file: str, image_key: str):
        with open(json_file, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data["data_list"]:
            path = entry[image_key]
            label = int(entry.get("gt_label", 0))
            samples.append((path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        rel_path, label = self.samples[index]
        img_path = os.path.join(self.root, rel_path) if self.root else rel_path
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
