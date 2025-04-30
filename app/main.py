import os, torch
import cv2, base64, numpy as np, time, gc, logging
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ─────────────  CONFIG  ─────────────
IM_SIZE     = 320          # 640 for max accuracy
CONF_TH     = 0.25
IOU_TH      = 0.6

CONNECTIONS = [(5, 6), (5, 11), (6, 12), (11, 12)]
# ────────────────────────────────────

app = FastAPI()
model: YOLO | None = None   # will be set in startup

# ----------  SCHEMAS ----------
class ImageRequest(BaseModel):
    id: str
    image: str               # base-64 JPEG/PNG

class DetectionResponse(BaseModel):
    id: str
    count: int
    boxes: List[dict]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float
# ------------------------------

# ----------  MODEL ----------
@app.on_event("startup")
def load_model() -> None:
    global model
    model = YOLO("./model1_yolox/yolo11x-pose.pt").to("cpu")
    model.fuse()
    model.overrides.update(
        imgsz=IM_SIZE, conf=CONF_TH, iou=IOU_TH,
        half=False, max_det=100, verbose=False, save=False
    )
    print(f"✅ model ready")
# ---------------------------

# ----------  HELPERS ----------
def decode_to_array(b64: str) -> np.ndarray:
    """base64 → BGR ndarray resized to IM_SIZE (no extra copies)."""
    img_bytes = base64.b64decode(b64)            # bytes
    np_view   = np.frombuffer(img_bytes, np.uint8)  # zero-copy
    img       = cv2.imdecode(np_view, cv2.IMREAD_COLOR)
    if img.shape[0] != IM_SIZE:
        img = cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
    return img                                    # BGR uint8

@torch.inference_mode()
def infer_one(img: np.ndarray):
    """Run model.predict with minimal allocations."""
    return model.predict(img, device="cpu", stream=False)[0]
# ------------------------------

def extract_detections(res):
    """Convert Ultralytics Results → (boxes, keypoints) JSONable."""
    boxes_xywh = res.boxes.xywh.cpu().numpy()     # (n,4)
    confs      = res.boxes.conf.cpu().numpy()     # (n,)
    kps_all    = res.keypoints.data.cpu().numpy() # (n,17,3)

    boxes = [
        {"x": float(cx - w/2), "y": float(cy - h/2),
         "width": float(w),    "height": float(h),
         "probability": float(p)}
        for (cx,cy,w,h), p in zip(boxes_xywh, confs)
    ]
    keypoints = kps_all.tolist()  # cheapest conversion

    return boxes, keypoints

# ----------  ENDPOINTS ----------
@app.post("/json_keypoints", response_model=DetectionResponse)
async def json_keypoints(req: ImageRequest):
    t0 = time.time()
    img = decode_to_array(req.image)
    preprocess_ms = (time.time() - t0) * 1e3

    t1 = time.time()
    res = infer_one(img)
    infer_ms = (time.time() - t1) * 1e3
    # immediately free img (large) & avoid copy retention
    del img

    t2 = time.time()
    boxes, keypoints = extract_detections(res) if len(res.boxes) else ([], [])
    post_ms = (time.time() - t2) * 1e3

    # drop Results object asap
    del res
    gc.collect()

    return {
        "id": req.id,
        "count": len(boxes),
        "boxes": boxes,
        "keypoints": keypoints,
        "speed_preprocess": round(preprocess_ms, 2),
        "speed_inference":  round(infer_ms, 2),
        "speed_postprocess": round(post_ms, 2),
    }

@app.post("/annotated_image")
async def annotated(req: ImageRequest):
    img = decode_to_array(req.image)
    res = infer_one(img)

    if not len(res.boxes):
        return {"id": req.id, "annotated_image": None}

    # ─── draw ───
    for kps, confs in zip(res.keypoints.xy, res.keypoints.conf):
        for i, (x, y) in enumerate(kps):
            if confs[i] > 0.5:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        for i, j in CONNECTIONS:
            if confs[i] > 0.5 and confs[j] > 0.5:
                cv2.line(img,
                         (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])),
                         (255, 0, 0), 1)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "JPEG encode failed")

    # convert first, then free
    encoded = base64.b64encode(buf).decode()

    del img, res, buf
    gc.collect()

    return {"id": req.id, "annotated_image": encoded}
# ---------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=60000, log_level="info", workers=2)