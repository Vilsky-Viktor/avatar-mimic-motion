import cv2, numpy as np
import torch
from retinaface import RetinaFace
from ultralytics import YOLO

def clamp(x1, y1, x2, y2, w, h):
    return max(0,int(x1)), max(0,int(y1)), min(int(x2),w-1), min(int(y2),h-1)

class FaceDetector:
    def detect(self, img):
        res = RetinaFace.detect_faces(img)

        if not isinstance(res, dict): return None

        best, area = None, -1

        for det in res.values():
            x1,y1,x2,y2 = det["facial_area"]
            a = (x2-x1)*(y2-y1)
            if a>area:
                area=a
                best={"box":(x1,y1,x2,y2),"landmarks":det["landmarks"]}

        return best

class BodyDetector:
    def __init__(self, weights="yolov8x-seg.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights).to(device)

    def detect(self, img):
        h,w=img.shape[:2]
        res=self.model.predict(img,verbose=False)[0]

        if res.masks is None: return None

        best,area=None,-1

        for i,(b,cls) in enumerate(zip(res.boxes.xyxy.cpu().numpy(),res.boxes.cls.cpu().numpy())):
            if int(cls)!=0: continue

            m=res.masks.data[i].cpu().numpy()
            mask=(cv2.resize(m,(w,h))>0.5).astype(np.uint8)
            a=int(mask.sum())

            if a>area:
                area=a; best={"box":clamp(*b,w,h),"mask":mask*255}
                
        return best