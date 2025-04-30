import base64, glob, json, os, random, uuid
from locust import HttpUser, task, between, events

# ---------- preload images ----------
IMG_DIR = "client/inputfolder"
IMG_B64 = [
    base64.b64encode(open(p, "rb").read()).decode()
    for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
]
if not IMG_B64:
    raise SystemExit(f"No .jpg files found in {IMG_DIR}")

print(f"Loaded {len(IMG_B64)} images into memory.")

# ---------- Locust user ----------
class PoseUser(HttpUser):
    host = "http://207.211.149.47:30003"    # <-- change if NodePort differs
    wait_time = between(0, 0)               # fire next request immediately

    def _payload(self):
        return {
            "id": str(uuid.uuid4()),
            "image": random.choice(IMG_B64)
        }

    @task(5)                                # 50 %
    def json_keypoints(self):
        self.client.post(
            "/json_keypoints",
            data=json.dumps(self._payload()),
            headers={"Content-Type": "application/json"},
            name="/json_keypoints",
            timeout=30
        )

    @task(5)                                # 50 %
    def annotated_image(self):
        self.client.post(
            "/annotated_image",
            data=json.dumps(self._payload()),
            headers={"Content-Type": "application/json"},
            name="/annotated_image",
            timeout=30
        )

# ---------- nice summary on quit ----------
@events.quitting.add_listener
def _(env, **kw):
    s = env.stats.get("/json_keypoints", "POST")
    if s:
        print(f"\nTotal requests : {s.num_requests}  "
              f"failures : {s.num_failures}  "
              f"avg {s.avg_response_time:.1f} ms  "
              f"p95 {s.get_response_time_percentile(0.95):.1f} ms")