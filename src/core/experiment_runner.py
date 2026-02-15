from src.acquisition.camera import CameraService
from src.acquisition.arduino import ArduinoSensor
from src.processing.image_features import extract_rgb_features
from src.database.repository import save_measurement
from src.database.backup import backup_database

camera = CameraService()
arduino = ArduinoSensor()


def run_capture_cycle(lectura_id=0, manual=False):

    filepath = camera.capture()

    r, g, b, i = extract_rgb_features(filepath)

    temp, ph = arduino.read()

    save_measurement(
        temp, ph, r, g, b, i,
        filepath,
        densidad_celular=0.0,
        lectura_id=lectura_id
    )

    backup_database()

    return filepath
