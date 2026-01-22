import cv2
import subprocess
import time
from collections import deque, Counter
from ultralytics import YOLO

# =========================
# YOLOv8 GESTURE DETECTOR
# =========================
class GestureDetectorYOLO:
    def __init__(self, model_path="gesture.pt", conf=0.5):
        """
        model_path : hasil training YOLOv8 gesture (gesture.pt)
        conf       : confidence threshold
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Return:
        - cls_id (int)
        - confidence (float)
        - bounding box (x1, y1, x2, y2)
        """
        results = self.model(frame, stream=True, verbose=False)
        best = None

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.conf:
                    if best is None or conf > best[1]:
                        best = (
                            int(box.cls[0]),
                            conf,
                            box.xyxy[0]
                        )

        return best if best else (None, None, None)


# =========================
# GESTURE STABILIZER
# =========================
class GestureRecognizer:
    def __init__(self):
        self.history = deque(maxlen=15)

    def stable(self, gesture):
        """
        Menghasilkan gesture stabil jika muncul >70%
        dalam window history
        """
        if gesture is None:
            return None

        self.history.append(gesture)

        if len(self.history) < 10:
            return None

        most, freq = Counter(self.history).most_common(1)[0]
        return most if freq / len(self.history) > 0.7 else None


# =========================
# MAIN APPLICATION (UCD)
# =========================
class GestureLauncher:
    def __init__(self):
        # GUNAKAN MODEL HASIL TRAINING
        self.detector = GestureDetectorYOLO(
            model_path="gesture.pt",
            conf=0.5
        )

        self.recognizer = GestureRecognizer()

        self.system_active = False
        self.last_action = 0
        self.cooldown = 3  # detik

        # HARUS SESUAI DENGAN data.yaml
        self.gesture_map = {
            0: "open_palm",
            1: "fist",
            2: "thumbs_up",
            3: "peace"
        }

        # Gesture → Aplikasi
        self.apps = {
            "thumbs_up": ("Google Chrome", "start chrome"),
            "peace": ("Spotify", "start spotify")
        }

    def launch_app(self, gesture):
        if time.time() - self.last_action < self.cooldown:
            return

        name, cmd = self.apps[gesture]
        subprocess.Popen(cmd, shell=True)
        self.last_action = time.time()
        print(f"✓ Launched {name}")

    def draw_status(self, frame, text, color):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        print("✋ HandLaunch YOLOv8 | Press Q to quit")

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cls, conf, box = self.detector.detect(frame)

            status_text = "System OFF - Show Open Palm"
            status_color = (0, 0, 255)

            if cls is not None:
                gesture = self.gesture_map.get(cls)
                if gesture is not None:
                    stable = self.recognizer.stable(gesture)

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame,
                                f"{gesture} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                    if stable:
                        # LOGIKA SISTEM (UCD)
                        if stable == "open_palm":
                            self.system_active = True
                        elif stable == "fist":
                            self.system_active = False
                        elif self.system_active and stable in self.apps:
                            self.launch_app(stable)

            if self.system_active:
                status_text = "System ACTIVE"
                status_color = (0, 255, 0)

            self.draw_status(frame, status_text, status_color)

            # FPS display
            cv2.putText(frame, f"FPS: {int(fps)}",
                        (frame.shape[1] - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            cv2.imshow("HandLaunch - YOLOv8", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    GestureLauncher().run()
