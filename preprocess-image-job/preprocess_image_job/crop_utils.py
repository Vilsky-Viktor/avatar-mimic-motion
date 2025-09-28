import cv2
import numpy as np

def pad_to_aspect(img, target_w=576, target_h=1024):
    """
    Pad image to target aspect ratio (default 9:16) with transparent padding.
    """
    # Ensure BGRA (with alpha)
    if img.shape[2] == 3:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Too wide → pad height
        new_h = int(w / target_aspect)
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left, pad_right = 0, 0
    else:
        # Too tall → pad width
        new_w = int(h * target_aspect)
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top, pad_bottom = 0, 0

    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)  # transparent
    )

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def crop_face(img, box, landmarks, target_w=576, target_h=1024, margin_factor=0.1):
    """
    Crop face region guided by landmarks, then pad to 9:16 aspect and resize.
    margin_factor adds a bit more context around the face.
    """
    x1, y1, x2, y2 = box
    le, re = landmarks["left_eye"], landmarks["right_eye"]
    ml, mr = landmarks["mouth_left"], landmarks["mouth_right"]

    # Compute vertical landmarks
    eye_y = (le[1] + re[1]) / 2
    mouth_y = (ml[1] + mr[1]) / 2
    em = max(8.0, mouth_y - eye_y)

    # Define vertical crop bounds
    top = int(eye_y - 1.2 * em)   # a bit more forehead
    bottom = int(mouth_y + 0.7 * em)  # include chin

    # Horizontal crop (centered on eyes/mouth line)
    cx = (x1 + x2) // 2
    half_w = int((x2 - x1) * (0.65 + margin_factor))

    h, w = img.shape[:2]
    top = max(0, top)
    bottom = min(h, bottom)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)

    crop = img[top:bottom, x1:x2]
    if crop.size == 0:
        return None

    return pad_to_aspect(crop, target_w, target_h)


def crop_body(img, box, margin=0.12, target_w=576, target_h=1024):
    """
    Crop body region with margin, then pad to 9:16 aspect and resize.
    margin ensures arms/legs aren’t cut off.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    # Expand box with margin
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return pad_to_aspect(crop, target_w, target_h)