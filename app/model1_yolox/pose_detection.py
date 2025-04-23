from ultralytics import YOLO
import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


def predict(model,src_img_path, dst_img_path):
    log.info(f"Predicting with source image: {src_img_path}, output to {dst_img_path}")

    # Read image efficiently using imdecode (assumes src_img_path is a valid file path)
    with open(src_img_path, 'rb') as f:
        img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)


    if img is None:
        log.error(f"Error: Could not read image at {src_img_path}")
        return None

    # Run inference
    results = model(img)

    for result in results:
        keypoints = result.keypoints

        if keypoints is not None and len(keypoints.xy) > 0:
            keypoints_xy = keypoints.xy[0]
            keypoints_conf = keypoints.conf[0]

            for k, (x, y) in enumerate(keypoints_xy):
                if keypoints_conf[k] > 0.5:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(img, str(k), (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            connections = [[5, 6], [5, 11], [6, 12], [11, 12]]
            for c in connections:
                if keypoints_conf[c[0]] > 0.5 and keypoints_conf[c[1]] > 0.5:
                    pt1 = tuple(map(int, keypoints_xy[c[0]]))
                    pt2 = tuple(map(int, keypoints_xy[c[1]]))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

            # Save the annotated image (optionally use lower quality to speed up saving)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Adjust quality if needed
            cv2.imwrite(dst_img_path, img, encode_param)
        else:
            log.warning("No keypoints detected in the image.")

    return results


def load_model():
    log.info("Loading YOLO pose detection model...")
    model = YOLO('./model1_yolox/yolo11x-pose.pt')
    log.info("Model is loaded.")
    return model

def main():
    try:
        log.info("Loading YOLO pose detection model...")
        model =load_model()
        # Load a pretrained YOLOv11x pose model
        log.info("Model is loaded.")
        # Predict on an image
        
        image_path = './model1_yolox/test.jpg'
        #image_path = './client/inputfolder/000000261033.jpg' 
        output_image = './model1_yolox/test_with_keypoints.jpg'
        result = predict(model,image_path,output_image)
        log.info(result)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()