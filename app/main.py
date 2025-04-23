from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO
import logging

from model1_yolox.pose_detection import predict

app = FastAPI()


# GLOBAL model variable
model = None


# Request model
class ImageRequest(BaseModel):
    id: str
    image: str

# Response model
class DetectionResponse(BaseModel):
    id: str
    count: int
    boxes: List[dict]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

def decode_base64_image(base64_str):
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)


@app.on_event("startup")
def load_model():
    global model
    model = YOLO("./model1_yolox/yolo11x-pose.pt")
    model.to("cpu")  # ensure model stays on CPU

@app.post("/json_keypoints", response_model=DetectionResponse)
async def get_pose_json(req: ImageRequest):
    global model

    preprocess_start = time.time()
    img = decode_base64_image(req.image)
    preprocess_end = time.time()

    inference_start = time.time()
    results = model(img)
    inference_end = time.time()

    post_start = time.time()
    result = results[0]

    print(f"Result: {result}")

    keypoints_all = []
    boxes = []

    try:
        # get bbox centers / sizes
        if hasattr(result.boxes.xywh, "cpu"):
            boxes_xywh = result.boxes.xywh.cpu().numpy()
        else:
            boxes_xywh = np.array(result.boxes.xywh)

        # get confidences
        if hasattr(result.boxes.conf, "cpu"):
            box_confs = result.boxes.conf.cpu().numpy()
        else:
            box_confs = np.array(result.boxes.conf)

        # raw keypoints tensor (n,keypoints,3)
        kps_np = result.keypoints.data.cpu().numpy()

        # build JSONables
        keypoints_all = [
            [[float(x), float(y), float(c)] for x, y, c in person_kps]
            for person_kps in kps_np
        ]

        boxes = []
        for (cx, cy, w, h), p in zip(boxes_xywh, box_confs):
            boxes.append({
                "x": float(cx - w/2),
                "y": float(cy - h/2),
                "width": float(w),
                "height": float(h),
                "probability": float(p)
            })

    except Exception as e:
        # now logs the real Python error instead of a mysterious 500
        raise HTTPException(status_code=500, detail=f"Post-processing error: {e}")

    post_end = time.time()

    return {
            "id": req.id,
            "count": len(boxes),
            "boxes": boxes,
            "keypoints": keypoints_all,
            "speed_preprocess": round(preprocess_end - preprocess_start, 4),
            "speed_inference": round(inference_end - inference_start, 4),
            "speed_postprocess": round(post_end - post_start, 4)
        }

connections = [
    (5, 6),   # left shoulder ↔ right shoulder
    (5, 11),  # left shoulder ↔ left hip
    (6, 12),  # right shoulder ↔ right hip
    (11, 12), # left hip ↔ right hip
]

@app.post("/annotated_image")
async def get_annotated_image(req: ImageRequest):
    img = decode_base64_image(req.image)
    results = model(img)
    result = results[0]

    # ensure there are keypoints to draw
    if not result.keypoints or result.keypoints.xy is None:
        raise HTTPException(status_code=200, detail={
            "id": req.id,
            "message": "No person detected",
            "annotated_image": None
        })

    # loop over each detected person
    for kps, confs in zip(result.keypoints.xy, result.keypoints.conf):
        # draw circles
        for idx, (x, y) in enumerate(kps):
            if confs[idx] > 0.5:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

        # draw connection lines
        for i, j in connections:
            if confs[i] > 0.5 and confs[j] > 0.5:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    # encode back to base64 JPEG
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    b64_img = base64.b64encode(buffer).decode('utf-8')

    return {"id": req.id, "annotated_image": b64_img}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=60000, log_level="info")