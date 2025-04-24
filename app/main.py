from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO
import logging
import gc

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

def decode_base64_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    # 3) Resize down to 320×320
    return cv2.resize(img, (320, 320))


@app.on_event("startup")
def load_model():
    global model
    model = YOLO("./model1_yolox/yolo11x-pose.pt")

@app.post("/json_keypoints", response_model=DetectionResponse)
async def get_pose_json(req: ImageRequest):
    # preprocess timing
    t0 = time.time()
    img = decode_base64_image(req.image)
    t1 = time.time()

    # inference timing (ultralytics already runs under no_grad())
    t2 = time.time()
    results = model(img)
    t3 = time.time()

    res = results[0]

    try:
        # ensure there are keypoints to process
        boxes_xywh = res.boxes.xywh.tolist()   # list of [cx, cy, w, h]
        box_confs  = res.boxes.conf.tolist()   # list of confidences
        kps_all    = res.keypoints.data.tolist()  # list of persons × keypoints × [x, y, c]

        # build JSON-friendly dicts
        boxes = [
            {
                "x": cx - w/2,
                "y": cy - h/2,
                "width": w,
                "height": h,
                "probability": p
            }
            for (cx, cy, w, h), p in zip(boxes_xywh, box_confs)
        ]

        keypoints = [
            [[x, y, c] for x, y, c in person_kps]
            for person_kps in kps_all
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-processing error: {e}")

    t4 = time.time()

    # 5) cleanup large temporaries immediately
    del results, res, boxes_xywh, box_confs, kps_all
    gc.collect()

    return {
        "id": req.id,
        "count": len(boxes),
        "boxes": boxes,
        "keypoints": keypoints,
        "speed_preprocess":   round((t1 - t0) * 1000, 2),
        "speed_inference":    round((t3 - t2) * 1000, 2),
        "speed_postprocess":  round((t4 - t3) * 1000, 2)
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