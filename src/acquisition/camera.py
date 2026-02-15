import cv2
import os
import time
from datetime import datetime

PHOTO_DIR = "photos"
os.makedirs(PHOTO_DIR, exist_ok=True)


class CameraService:

    def capture(self) -> str:

        camera = cv2.VideoCapture(0)

        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera.set(cv2.CAP_PROP_FOCUS, 65)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)
        camera.set(cv2.CAP_PROP_EXPOSURE, 70)

        if not camera.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara")

        time.sleep(2)

        ret = False
        for _ in range(10):
            ret, frame = camera.read()
            if ret:
                break
            time.sleep(0.1)

        if not ret:
            camera.release()
            raise RuntimeError("No se pudo capturar imagen")

        now = datetime.now()
        subdir = os.path.join(
            PHOTO_DIR,
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d"),
        )

        os.makedirs(subdir, exist_ok=True)

        filename = f"photo_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(subdir, filename)

        cv2.imwrite(filepath, frame)
        camera.release()

        return filepath
