import os, json
from pathlib import Path
from preprocess_image_job.gcs_utils import get_client, read_json, list_images, download_image, upload_png
from preprocess_image_job.detectors import FaceDetector, BodyDetector
from preprocess_image_job.crop_utils import crop_face, crop_body
from preprocess_image_job.sam2_utils import Sam2Wrapper
import logging

BUCKET=os.environ["JOBS_BUCKET"]
EXEC_ID=os.environ["EXECUTION_ID"]

def main():
    client=get_client()
    bucket=client.bucket(BUCKET)
    manifest=read_json(bucket,f"jobs/{EXEC_ID}/manifest.json")
    aid=manifest["ai_avatar_id"]
    in_prefix=f"ai_avatars/{aid}/identity_images/"
    out_root=f"ai_avatars/{aid}/dataset/"
    imgs=list_images(bucket,in_prefix)
    face_det,body_det=FaceDetector(),BodyDetector()
    sam2=Sam2Wrapper(bucket)

    reports=[]
    for path in imgs:
        logging.info(f"processing image {path}")
        img=download_image(bucket,path); name=Path(path).stem
        face=face_det.detect(img); body=body_det.detect(img)
        used=False; reasons=[]

        if face:
            logging.info("face detected ... cropping")
            fc=crop_face(img,face["box"],face["landmarks"],224)
            
            if fc is not None:
                upload_png(bucket,f"{out_root}face_crops/{name}.png",fc)
                alpha=sam2.alpha_from_box(img,face["box"],224)
                upload_png(bucket,f"{out_root}mattes/{name}_face_alpha.png",alpha)
                used=True
            else: 
                reasons.append("face_crop_failed")
                logging.error("face crop failed")

        else: 
            reasons.append("face_not_detected")
            logging.warn("face not detected")

        if body:
            logging.info("body detected ... cropping")
            bc=crop_body(img,body["box"],0.08,512)

            if bc is not None:
                upload_png(bucket,f"{out_root}body_crops/{name}.png",bc)
                alpha=sam2.alpha_from_box(img,body["box"],512)
                upload_png(bucket,f"{out_root}mattes/{name}_body_alpha.png",alpha)
                used=True

            else: 
                reasons.append("body_crop_failed")
                logging.error("body crop failed")

        else: 
            reasons.append("body_not_detected")
            logging.warn("body not detected")

        reports.append({"img":path,"used":used,"reasons":reasons})

    logging.info("composing metadata")
    meta={"execution_id":EXEC_ID,"ai_avatar_id":aid,"images":reports}
    bucket.blob(f"{out_root}metadata.json").upload_from_string(json.dumps(meta,indent=2),"application/json")
    logging.info("[DONE]")

if __name__=="__main__":
    main()