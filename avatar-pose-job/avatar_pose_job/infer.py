import json
from pathlib import Path
from PIL import Image
import torch

# DWPose repo detector
from mmpose.apis import init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples

CONFIG_FILE = "/app/DWPose/mmpose_configs/dwpose/dwpose-l_384x288.py"
CHECKPOINT_FILE = "/app/DWPose/dwpose-l.pth"

def run_dwpose_on_dir(photos_dir: Path, out_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_pose_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(photos_dir.glob("*.[jp][pn]g")):
        image = Image.open(img_path).convert("RGB")

        # Run inference
        result = inference_topdown(model, str(img_path))
        merged = merge_data_samples(result)

        keypoints = merged.pred_instances.keypoints[0].tolist()  # [N,2]
        scores = merged.pred_instances.keypoint_scores[0].tolist()  # [N]

        data = {"file": img_path.name, "keypoints": keypoints, "scores": scores}
        out_file = out_dir / f"{img_path.stem}.json"

        with open(out_file, "w") as f:
            json.dump(data, f)

        print(f"[DWPose] {img_path.name} -> {out_file}")