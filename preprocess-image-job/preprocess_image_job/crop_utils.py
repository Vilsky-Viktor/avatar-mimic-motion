import cv2

def pad_to_square(img, target_size):
    h,w=img.shape[:2]
    side=max(h,w)
    top=(side-h)//2; bottom=side-h-top
    left=(side-w)//2; right=side-w-left
    img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))

    return cv2.resize(img,(target_size,target_size),interpolation=cv2.INTER_LINEAR)

def crop_face(img, box, landmarks, size=224):
    x1, y1, x2, y2 = box
    le, re = landmarks["left_eye"], landmarks["right_eye"]
    ml, mr = landmarks["mouth_left"], landmarks["mouth_right"]

    eye_y   = (le[1] + re[1]) / 2
    mouth_y = (ml[1] + mr[1]) / 2
    em      = max(8.0, mouth_y - eye_y)

    top    = int(eye_y - 1.2 * em)
    bottom = int(mouth_y + 0.7 * em)

    cx = (x1 + x2) // 2
    half_w = int((x2 - x1) * 0.65)

    h, w = img.shape[:2]
    top    = max(0, top)
    bottom = min(h, bottom)
    x1     = max(0, cx - half_w)
    x2     = min(w, cx + half_w)

    crop = img[top:bottom, x1:x2]
    if crop.size == 0:
        return None

    return pad_to_square(crop, size)

def crop_body(img, box, margin=0.1, size=512):
    h,w=img.shape[:2]; x1,y1,x2,y2=box
    bw, bh=x2-x1, y2-y1
    mx, my=int(bw*margin), int(bh*margin)
    x1,y1=max(0,x1-mx),max(0,y1-my)
    x2,y2=min(w,x2+mx),min(h,y2+my)
    crop=img[y1:y2,x1:x2]

    if crop.size==0: return None
    
    return pad_to_square(crop,size)