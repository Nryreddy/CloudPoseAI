# main.py – faster CPU inference (same outputs)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import base64, cv2, numpy as np, time, gc, logging, os, torch
from ultralytics import YOLO

app = FastAPI()

# ────────── Hyper-parameters you can still tune ──────────
IM_SIZE   = 320            # keep 320 for speed; 640 if you want max accuracy
CONF_TH   = 0.25           # det-confidence threshold
IOU_TH    = 0.6            # NMS IoU
NUM_THREADS = max(1, os.cpu_count() - 1)   # leave 1 core for the OS
# ─────────────────────────────────────────────────────────

torch.set_num_threads(NUM_THREADS)

# COCO keypoint edges you like to draw
CONNECTIONS = [(5, 6), (5, 11), (6, 12), (11, 12)]

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

# --------------------------------------------------
#                MODEL  INIT
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    # 1. load the big model once on CPU
    model = YOLO("./model1_yolox/yolo11x-pose.pt").to("cpu")
    # 2. fuse Conv+BN layers for faster matmul
    model.fuse()
    # 3. set default inference args so we don’t repeat them
    model.overrides.update(
        imgsz=IM_SIZE,
        conf=CONF_TH,
        iou=IOU_TH,
        half=False,          # CPU can’t run fp16
        max_det=100,
        verbose=False
    )
    print(f"Model loaded, fused, ready on CPU ({NUM_THREADS} threads).")

# --------------------------------------------------
#                UTILITIES
# --------------------------------------------------
def decode_and_resize(b64: str) -> np.ndarray:
    """base64 → BGR ndarray resized to IM_SIZE."""
    img_arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)

@torch.inference_mode()     # guarantees no grad / no autograd buffers
def run_inference(img: np.ndarray):
    """Fastest possible Ultralytics call on CPU."""
    results = model.predict(
        img,                 # ndarray
        device="cpu",
        stream=False         # list, not generator
    )
    return results[0]       # single image result

# --------------------------------------------------
#          POST-PROCESS helpers (unchanged)
# --------------------------------------------------
def extract_detections(res):
    boxes_xywh = res.boxes.xywh.tolist()
    box_confs  = res.boxes.conf.tolist()
    kps_all    = res.keypoints.data.tolist()

    boxes = [
        {"x": cx - w/2, "y": cy - h/2, "width": w, "height": h, "probability": p}
        for (cx, cy, w, h), p in zip(boxes_xywh, box_confs)
    ]
    keypoints = [
        [[x, y, c] for x, y, c in person_kps] for person_kps in kps_all
    ]
    return boxes, keypoints

def annotate_image(img, res):
    for kps, confs in zip(res.keypoints.xy, res.keypoints.conf):
        for i, (x, y) in enumerate(kps):
            if confs[i] > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        for i, j in CONNECTIONS:
            if confs[i] > 0.5 and confs[j] > 0.5:
                p1 = (int(kps[i][0]), int(kps[i][1]))
                p2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(img, p1, p2, (255, 0, 0), 1)
    return img

# --------------------------------------------------
#            FASTAPI  ENDPOINTS
# --------------------------------------------------
@app.post("/json_keypoints", response_model=DetectionResponse)
async def get_pose_json(req: ImageRequest):
    t0 = time.time()
    img = decode_and_resize(req.image)
    t_pre = (time.time() - t0) * 1000

    t1 = time.time()
    res = run_inference(img)
    t_inf = (time.time() - t1) * 1000

    t2 = time.time()
    boxes, keypoints = extract_detections(res) if res.boxes else ([], [])
    t_post = (time.time() - t2) * 1000
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
    res = run_inference(img)
    if not res.boxes:
        return {"id": req.id, "annotated_image": None}

    annotated = annotate_image(img, res)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")

    return {"id": req.id, "annotated_image": base64.b64encode(buf).decode()}

# --------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn_config = dict(host="0.0.0.0", port=60000, log_level="info", workers=1)
    import uvicorn
    uvicorn.run(app, **uvicorn_config)
