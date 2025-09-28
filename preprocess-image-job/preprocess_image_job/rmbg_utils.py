from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import cv2
import numpy as np


class RMBG2:
    def __init__(self, model_dir="/models/rmbg2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RMBG-2.0] Loading model from {model_dir} on {self.device}")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_dir,
            trust_remote_code=True
        ).eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def remove_bg(self, bgr_img: np.ndarray):
        """
        Apply background removal (same as official RMBG-2.0 example).
        Returns:
            rgba (np.ndarray): Image with alpha channel applied
        """
        # Convert cv2 BGR → PIL RGB
        image = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        # Resize mask back to original size & add alpha
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)

        # Convert back to cv2 RGBA
        rgba = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA)
        return rgba