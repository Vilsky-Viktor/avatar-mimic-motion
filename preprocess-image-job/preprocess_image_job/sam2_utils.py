import cv2, numpy as np
import torch
from pathlib import Path
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "sam2.1_hiera_large.pt")

CHECKPOINT_TO_CONFIG = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
}


class Sam2Wrapper:
    def __init__(self, bucket, ckpt_dir="/models/sam2"):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Target checkpoint
        ckpt_path = self.ckpt_dir / SAM_MODEL_NAME

        # Download only the requested model if not already present
        if not ckpt_path.exists():
            blob = bucket.blob(f"models/sam2/{SAM_MODEL_NAME}")
            if not blob.exists():
                raise FileNotFoundError(f"{SAM_MODEL_NAME} not found in bucket")
            print(f"downloading {blob.name}")
            blob.download_to_filename(str(ckpt_path))

        ckpt_key = ckpt_path.stem
        if ckpt_key not in CHECKPOINT_TO_CONFIG:
            raise ValueError(f"No config mapping for checkpoint {ckpt_key}")

        config_file = f"configs/sam2.1/{CHECKPOINT_TO_CONFIG[ckpt_key]}"

        print(f"[SAM2] Using config={config_file}, ckpt={ckpt_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")

        model = build_sam2(config_file=config_file, ckpt_path=str(ckpt_path), device=device)
        self.predictor = SAM2ImagePredictor(model)
        
    def alpha_from_box(self,img,bbox,size):
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        x1,y1,x2,y2=bbox
        masks,_,_=self.predictor.predict(box=np.array([[x1,y1,x2,y2]],dtype=np.float32))
        m=(masks[0]>0.5).astype(np.uint8)*255
        m=cv2.resize(m,(size,size),interpolation=cv2.INTER_NEAREST)
        
        return m