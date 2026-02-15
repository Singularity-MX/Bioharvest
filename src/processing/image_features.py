import cv2
import numpy as np


def extract_rgb_features(filepath):

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    roi = image[
        int((h-250)/2):int((h+250)/2),
        int((w-250)/2):int((w+250)/2)
    ]

    blue, green, red = cv2.split(roi)
    intensity = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    return (
        float(np.mean(red)),
        float(np.mean(green)),
        float(np.mean(blue)),
        float(np.mean(intensity)),
    )
