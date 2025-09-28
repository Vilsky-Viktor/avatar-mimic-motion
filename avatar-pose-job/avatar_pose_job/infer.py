import json
from pathlib import Path
import cv2
import numpy as np
from dwpose import DwposeDetector

# LoRA target size (9:16)
FINAL_W, FINAL_H = 576, 1024

def extract_keypoints(result):
    """Extract all 133 keypoints (body+face+hands) from dwpose result dict."""
    if not result or "people" not in result or not result["people"]:
        return []

    person = result["people"][0]
    all_points = []

    for k in ["pose_keypoints_2d", "face_keypoints_2d",
              "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
        pts = person.get(k)
        if not pts:
            continue
        for i in range(0, len(pts), 3):
            x, y, c = pts[i:i+3]
            all_points.append([x, y, c])

    return all_points

def run_dwpose_on_dir(photos_dir: Path, out_dir: Path,
                      include_hands=True, include_face=True):
    dw = DwposeDetector.from_pretrained_default()
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(photos_dir.glob("*.[jp][pn]g")):
        print(f"[DWPose] Processing {img_path.name}")

        # Load LoRA crop directly (576×1024)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️ Skipping {img_path}, not found")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run DWPose directly on full resolution
        pose_img, result, _ = dw(
            img_rgb,
            include_hand=include_hands,
            include_face=include_face,
            image_and_json=True,
            detect_resolution=1024  # match long side
        )

        # Extract keypoints
        keypoints = extract_keypoints(result)
        keypoints = keypoints.tolist() if hasattr(keypoints, "tolist") else keypoints

        # Normalize coordinates to [0,1]
        keypoints_normalized = [
            {"x": x / FINAL_W, "y": y / FINAL_H, "score": c}
            for x, y, c in keypoints
        ]

        # Save debug overlay (already same size as input)
        pose_img_np = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
        out_pose = out_dir / f"{img_path.stem}_pose.png"
        cv2.imwrite(str(out_pose), pose_img_np)

        # Save JSON
        out_json = out_dir / f"{img_path.stem}.json"
        with open(out_json, "w") as f:
            json.dump({
                "file": img_path.name,
                "size": [FINAL_W, FINAL_H],
                "skeleton": "dwpose_ucoco_133",
                "keypoints": keypoints_normalized
            }, f, indent=2)

        print(f"[DWPose] Saved {out_json}, {out_pose}")