from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import base64
import cv2
import numpy as np
import uuid
import logging
import time
import os
from typing import Dict
from model1_yolox.pose_detection import predict  # Import from your pose detection module
from client.cloudpose_client import call_cloudpose_service, get_images_to_be_processed  # Import client functions

app = FastAPI()
log = logging.getLogger(__name__)

# Initialize model at startup
model = None
MODEL_PATH = os.path.join("app", "model1-yolox", "yolo11x-pose.pt")

@app.on_event("startup")
async def load_model():
    global model
    try:
        from ultralytics import YOLO
        log.info("Loading YOLO pose detection model...")
        model = YOLO(MODEL_PATH)
        log.info("Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise

async def process_image_base64(image_base64: str) -> Dict:
    """Process base64 encoded image using existing pose detection logic"""
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        # Save temp image for processing (using existing predict function)
        temp_input = "temp_input.jpg"
        temp_output = "temp_output.jpg"
        cv2.imwrite(temp_input, img)
        
        # Use existing predict function
        results = predict(model, temp_input, temp_output)
        
        # Process results into required format
        keypoints_data = []
        boxes_data = []
        
        for result in results:
            if result.keypoints is not None:
                for person_idx in range(len(result.keypoints.xy)):
                    # Keypoints processing
                    kpts = result.keypoints.xy[person_idx].cpu().numpy()
                    confs = result.keypoints.conf[person_idx].cpu().numpy()
                    person_kpts = [[float(x), float(y), float(conf)] 
                                 for (x, y), conf in zip(kpts, confs)]
                    keypoints_data.append(person_kpts)
                    
                    # Boxes processing
                    if len(result.boxes) > person_idx:
                        box = result.boxes[person_idx]
                        boxes_data.append({
                            "x": float(box.xywh[0][0].item()),
                            "y": float(box.xywh[0][1].item()),
                            "width": float(box.xywh[0][2].item()),
                            "height": float(box.xywh[0][3].item()),
                            "probability": float(box.conf.item())
                        })
        
        # Read annotated image if needed
        annotated_img = cv2.imread(temp_output)
        
        # Cleanup temp files
        os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
        return {
            "count": len(keypoints_data),
            "boxes": boxes_data,
            "keypoints": keypoints_data,
            "annotated_image": annotated_img
        }
        
    except Exception as e:
        log.error(f"Image processing failed: {e}")
        raise

@app.post("/pose/json")
async def pose_json_api(data: Dict):
    """Endpoint matching the original client API specification"""
    try:
        if "id" not in data or "image" not in data:
            raise HTTPException(status_code=400, detail="Missing id or image in request")
        
        start_time = time.time()
        result = await run_in_threadpool(process_image_base64, data["image"])
        elapsed_time = time.time() - start_time
        
        return {
            "id": data["id"],
            "count": result["count"],
            "boxes": result["boxes"],
            "keypoints": result["keypoints"],
            "speed_preprocess": 0,  # Modify based on actual timing
            "speed_inference": elapsed_time,
            "speed_postprocess": 0   # Modify based on actual timing
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pose/image")
async def pose_image_api(data: Dict):
    """Endpoint for returning annotated images"""
    try:
        if "id" not in data or "image" not in data:
            raise HTTPException(status_code=400, detail="Missing id or image in request")
        
        result = await run_in_threadpool(process_image_base64, data["image"])
        
        if result["annotated_image"] is None:
            raise HTTPException(status_code=404, detail="No keypoints detected")
        
        _, buffer = cv2.imencode('.jpg', result["annotated_image"])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "id": data["id"],
            "image": encoded_image
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_process")
async def batch_process(input_folder: str, num_workers: int = 4):
    """Endpoint that uses the original client processing logic"""
    try:
        images = get_images_to_be_processed(input_folder)
        results = []
        
        def process_wrapper(image):
            try:
                with open(image, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                return call_cloudpose_service({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_OID, image)),
                    "image": img_base64
                })
            except Exception as e:
                log.error(f"Failed to process {image}: {e}")
                return None
                
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_wrapper, images))
        
        return {
            "processed": len([r for r in results if r is not None]),
            "failed": len([r for r in results if r is None]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=60000)