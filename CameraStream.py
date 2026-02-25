import threading
import cv2
import subprocess
import time


class CameraStream:
    def __init__(self, src=0, width=320, height=240, fps=30):
        subprocess.run(["fuser", "-k", "/dev/video0"], capture_output=True)
        time.sleep(1)

        # Lock exposure for consistent 30 FPS regardless of lighting
        subprocess.run(
            [
                "v4l2-ctl",
                "--device=/dev/video0",
                "--set-ctrl=exposure_dynamic_framerate=0",
            ]
        )
        subprocess.run(
            ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=auto_exposure=1"]
        )
        subprocess.run(
            ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=gain=120"]
        )  # Amplifies the signal (Add brightness, grains/noise)
        subprocess.run(
            ["v4l2-ctl", "--device=/dev/video0", "--set-ctrl=brightness=160"]
        )  # Shifts pixel values up/down

        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Drain stale buffer frames
        for _ in range(10):
            self.cap.read()

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def get(self, prop):
        return self.cap.get(prop)

    def getBackendName(self):
        return self.cap.getBackendName()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
