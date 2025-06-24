# CloudPose – Scalable Pose-Estimation API on Kubernetes
CloudPose is a small but production ready web service that turns any JPEG into structured human-pose data.  
It is built with **FastAPI** + **Ultralytics YOLOv11x** for CPU inference, wrapped in a small Docker image, and deployed as a horizontally scalable Kubernetes deployment.

![image](https://github.com/user-attachments/assets/437632c6-5899-409d-b61e-e12ce9c98fea)
![image](https://github.com/user-attachments/assets/05322223-2587-4ee2-9085-5be313ef77db)

## Why this project?
* **Two clean REST endpoints**  
  * `/json_keypoints` → returns key-points & boxes as JSON  
  * `/annotated_image` → returns the same image with a skeleton overlay
*  **Multi-stage Dockerfile** – final image < 750 MB
*  **Kubernetes ready** – includes deployment & NodePort service YAML  
  Scales from 1 → 8 pods (0.5 CPU / 512 MiB each) with automatic health probes
*  **Load-tested with Locust** – experiment scripts + results table & plots


## Quick start

```bash
git clone https://github.com/Nryreddy/CloudPoseAI.git
cd cloudpose
docker build -t CloudPoseAI-api .
docker run -p 60000:60000 CloudPoseAI-api
# open swagger: http://localhost:60000/docs
