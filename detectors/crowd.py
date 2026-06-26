import cv2
import numpy as np
import base64

CONFIDENCE_THRESHOLD = 0.4
CROWD_THRESHOLD = 10


class CrowdDetector:

    def __init__(self, model):
        # Shares the same yolov8n model — no separate load needed
        self.model = model

    def run(self, frame, all_rois=None, prev_gray=None):
        results = self.model(frame, verbose=False)
        person_count = 0
        display = frame.copy()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        crowd_detected = person_count >= CROWD_THRESHOLD

        if crowd_detected:
            cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 200), -1)
            cv2.putText(display, "CROWD ALERT — " + str(person_count) + " People",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(display, "People Count: " + str(person_count),
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if crowd_detected else (0, 255, 0), 3)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "status": "success",
            "crowd_detected": crowd_detected,
            "person_count": person_count,
            "human_detected": False,
            "fire_detected": False,
            "motion_detected": False,
            "details": "Crowd detected — " + str(person_count) + " people" if crowd_detected else "",
            "annotated_image": ann_b64
        }