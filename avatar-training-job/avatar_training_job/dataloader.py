import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch

class PoseImageDataset(Dataset):
    def __init__(self, photos_dir: Path, poses_dir: Path, image_size: int = 256):
        self.images = sorted(list(photos_dir.glob("*.jpg")) + list(photos_dir.glob("*.png")))
        self.poses_dir = poses_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pose_path = self.poses_dir / (img_path.stem + ".json")

        image = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(image)

        # Load DWPose keypoints JSON
        with open(pose_path, "r") as f:
            data = json.load(f)

        # Simplified: flatten keypoints (x,y,confidence)
        keypoints = torch.tensor(data["keypoints"], dtype=torch.float32)  # [N,3]
        keypoints = keypoints.flatten()  # [N*3]

        return tensor_img, keypoints