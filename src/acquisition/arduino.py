import serial
import json
import time
import os


class ArduinoSensor:

    def read(self):

        puerto = os.getenv("ARDUINO_PORT", "/dev/ttyACM0")

        ser = serial.Serial(puerto, 9600, timeout=1)
        time.sleep(10)

        ser.write(b"d\n")
        respuesta = ser.readline().decode().strip()

        try:
            datos = json.loads(respuesta)
            return datos.get("t"), datos.get("ph")

        except json.JSONDecodeError:
            return None, None

        finally:
            ser.close()
