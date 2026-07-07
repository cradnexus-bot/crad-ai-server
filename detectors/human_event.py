import cv2
import numpy as np
import base64
import time
from PIL import Image
import io

CONFIDENCE_THRESHOLD = 0.4
GRACE_FRAMES = 2
GIF_MAX_FRAMES = 12
BOX_COLOR = (0, 255, 0)
LABEL_COLOR = (255, 255, 255)


class HumanEventDetector:

    def __init__(self, model):
        self.model = model
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "in_event": False,
                "event_start_time": None,
                "event_frames": [],
                "event_max_count": 0,
                "event_position": "",
                "no_human_streak": 0,
            }
        return self.sessions[session_id]

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_position(self, boxes, w, h):
        if not boxes:
            return "unknown"
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        vert = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "center")
        horiz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
        return vert + "-" + horiz

    def draw_boxes(self, img, boxes):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 3)
            label = "Human " + str(round(conf, 2))
            cv2.putText(out, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1)
        return out

    def make_gif_b64(self, frames):
        if not frames:
            return ""
        try:
            pil_frames = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
            buf = io.BytesIO()
            pil_frames[0].save(
                buf, format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,
                duration=500,
                optimize=False
            )
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print("GIF error: " + str(e))
            return ""

    def run(self, frame, session_id, base_file_name="", pc_name=""):
        state = self.get_session(session_id)
        results = self.model(frame, verbose=False, classes=[0],
                             conf=CONFIDENCE_THRESHOLD)[0]
        boxes = results.boxes
        human_detected = len(boxes) > 0
        human_count = len(boxes)

        annotated = self.draw_boxes(frame.copy(), boxes) if human_detected else frame.copy()
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        event_data = None

        if human_detected:
            state["no_human_streak"] = 0

            if not state["in_event"]:
                state["in_event"] = True
                state["event_start_time"] = time.time()
                state["event_frames"] = []
                state["event_max_count"] = 0
                state["event_position"] = ""

            if len(state["event_frames"]) < GIF_MAX_FRAMES:
                state["event_frames"].append(annotated.copy())

            if human_count > state["event_max_count"]:
                state["event_max_count"] = human_count
                h, w = frame.shape[:2]
                state["event_position"] = self.get_position(boxes, w, h)

        else:
            if state["in_event"]:
                state["no_human_streak"] += 1
                if state["no_human_streak"] >= GRACE_FRAMES:
                    end_time = time.time()
                    duration = round(end_time - state["event_start_time"], 1)
                    gif_b64 = self.make_gif_b64(state["event_frames"])

                    event_data = {
                        "base_file_name": base_file_name,
                        "human_start_time": state["event_start_time"],
                        "human_end_time": end_time,
                        "duration": duration,
                        "position": state["event_position"],
                        "human_count": state["event_max_count"],
                        "gif_b64": gif_b64,
                        "pc_name": pc_name,
                        "status": "Detected"
                    }

                    state["in_event"] = False
                    state["event_start_time"] = None
                    state["event_frames"] = []
                    state["event_max_count"] = 0
                    state["event_position"] = ""
                    state["no_human_streak"] = 0

        return {
            "status": "success",
            "human_detected": human_detected,
            "human_count": human_count,
            "event_data": event_data,
            "annotated_image": ann_b64 if human_detected else ""
        }