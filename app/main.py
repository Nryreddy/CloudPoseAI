from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Any
import base64, cv2, numpy as np, time, gc, logging
from ultralytics import YOLO

app = FastAPI()
model = None

# your COCO‐style connections
CONNECTIONS = [(5,6), (5,11), (6,12), (11,12)]

class ImageRequest(BaseModel):
    id: str
    image: str

class DetectionResponse(BaseModel):
    id: str
    count: int
    boxes: List[dict]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

@app.on_event("startup")
def load_model():
    global model
    model = YOLO("./model1_yolox/yolo11x-pose.pt")

def decode_and_resize(b64: str, size=(320,320)) -> np.ndarray:
    data = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, size)

def run_inference(img: np.ndarray) -> Tuple[Any, float]:
    t0 = time.time()
    results = model(img)
    return results[0], (time.time() - t0) * 1000

def extract_detections(res) -> Tuple[List[dict], List[List[List[float]]]]:
    # Ultralytics already gives Python lists, so no .cpu().numpy() copies
    boxes_xywh = res.boxes.xywh.tolist()
    box_confs  = res.boxes.conf .tolist()
    kps_all    = res.keypoints.data.tolist()

    boxes = [
        {"x": cx - w/2, "y": cy - h/2, "width": w, "height": h, "probability": p}
        for (cx, cy, w, h), p in zip(boxes_xywh, box_confs)
    ]

    keypoints = [
        [[x, y, c] for x,y,c in person_kps]
        for person_kps in kps_all
    ]

    return boxes, keypoints

def annotate_image(img: np.ndarray, res) -> np.ndarray:
    for kps, confs in zip(res.keypoints.xy, res.keypoints.conf):
        for idx, (x, y) in enumerate(kps):
            if confs[idx] > 0.5:
                cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)
        for i,j in CONNECTIONS:
            if confs[i] > 0.5 and confs[j] > 0.5:
                p1, p2 = (int(kps[i][0]), int(kps[i][1])), (int(kps[j][0]), int(kps[j][1]))
                cv2.line(img, p1, p2, (255,0,0), 2)
    return img

@app.post("/json_keypoints", response_model=DetectionResponse)
async def get_pose_json(req: ImageRequest):
    # 1) decode
    t0 = time.time()
    img = decode_and_resize(req.image)
    t_pre = (time.time() - t0) * 1000

    # 2) inference
    res, t_inf = run_inference(img)

    # 3) post-process
    t1 = time.time()
    try:
        boxes, keypoints = extract_detections(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-processing error: {e}")
    t_post = (time.time() - t1) * 1000

    # 4) cleanup
    gc.collect()

    return {
        "id": req.id,
        "count": len(boxes),
        "boxes": boxes,
        "keypoints": keypoints,
        "speed_preprocess": round(t_pre, 2),
        "speed_inference":  round(t_inf, 2),
        "speed_postprocess": round(t_post, 2),
    }

@app.post("/annotated_image")
async def get_annotated_image(req: ImageRequest):
    img = decode_and_resize(req.image)
    res, _ = run_inference(img)

    if not res.keypoints or res.keypoints.xy is None:
        raise HTTPException(status_code=200, detail={
            "id": req.id, "message": "No person detected", "annotated_image": None
        })

    annotated = annotate_image(img, res)
    success, buf = cv2.imencode('.jpg', annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return {"id": req.id, "annotated_image": base64.b64encode(buf).decode('utf-8')}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=60000, log_level="info")