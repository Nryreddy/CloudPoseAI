from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO
import logging

app = FastAPI()

# Load model once at startup
model = YOLO("./model1_yolox/yolo11x-pose.pt")

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

@app.post("/json_keypoints", response_model=DetectionResponse)
async def get_pose_json(req: ImageRequest):
    preprocess_start = time.time()
    img = decode_base64_image(req.image)
    preprocess_end = time.time()

    inference_start = time.time()
    results = model(img)
    inference_end = time.time()

    post_start = time.time()
    result = results[0]
    keypoints_all = []
    boxes = []

    for kps, box in zip(result.keypoints.xy, result.boxes):
        kps_array = kps.tolist()
        confs = result.keypoints.conf[0].tolist()
        keypoints_all.append([[float(x), float(y), float(c)] for (x, y), c in zip(kps_array, confs)])

        b = box.xywh[0]
        boxes.append({
            "x": float(b[0] - b[2] / 2),
            "y": float(b[1] - b[3] / 2),
            "width": float(b[2]),
            "height": float(b[3]),
            "probability": float(box.conf[0])
        })
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

@app.post("/annotated_image")
async def get_annotated_image(req: ImageRequest):
    img = decode_base64_image(req.image)
    results = model(img)
    result = results[0]

    if result.keypoints.xy is not None:
        for kps, confs in zip(result.keypoints.xy, result.keypoints.conf):
            for i, (x, y) in enumerate(kps):
                if confs[i] > 0.5:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    _, buffer = cv2.imencode('.jpg', img)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return {"id": req.id, "annotated_image": b64_img}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=60000, log_level="info")