import cv2
import numpy as np
import base64
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5

VIOLATION_CLASSES = {"no-hardhat", "no-mask", "no-safety vest"}


class PPEDetector:

    def __init__(self, model_path):
        print("Loading PPE model...")
        self.model = YOLO(model_path)
        print("PPE model loaded.")

    def run(self, frame, all_rois=None, prev_gray=None):
        results = self.model(frame, verbose=False)
        violation_found = False
        alerted = set()
        details_parts = []
        display = frame.copy()

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                class_lower = class_name.lower()
                color = (0, 255, 0)

                if class_lower == "no-hardhat":
                    color = (0, 0, 255)
                    if class_lower not in alerted:
                        violation_found = True
                        details_parts.append("No Hardhat conf=" + str(round(confidence, 2)))
                        alerted.add(class_lower)
                elif class_lower == "no-mask":
                    color = (0, 0, 200)
                    if class_lower not in alerted:
                        violation_found = True
                        details_parts.append("No Mask conf=" + str(round(confidence, 2)))
                        alerted.add(class_lower)
                elif class_lower == "no-safety vest":
                    color = (0, 165, 255)
                    if class_lower not in alerted:
                        violation_found = True
                        details_parts.append("No Safety Vest conf=" + str(round(confidence, 2)))
                        alerted.add(class_lower)
                elif class_lower == "hardhat":
                    color = (0, 255, 0)
                elif class_lower == "mask":
                    color = (0, 200, 0)
                elif class_lower == "safety vest":
                    color = (255, 180, 0)
                elif class_lower == "person":
                    color = (200, 200, 200)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, class_name + " " + str(round(confidence, 2)),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        if violation_found:
            cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 200), -1)
            cv2.putText(display, "PPE VIOLATION DETECTED",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 3)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "status": "success",
            "violation_detected": violation_found,
            "human_detected": False,
            "fire_detected": False,
            "crowd_detected": False,
            "motion_detected": False,
            "details": " | ".join(details_parts),
            "annotated_image": ann_b64
        }