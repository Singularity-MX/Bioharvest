import pandas as pd
import numpy as np
import random
import cv2
import os
from pathlib import Path


class ImageRobustnessTool:

    # ----------------------------------------------------
    # INIT
    # ----------------------------------------------------
    def __init__(self, photos_dir, dataset_csv, output_dir):

        self.photos_dir = Path(photos_dir)
        self.dataset_csv = Path(dataset_csv)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\nInicializando herramienta...")
        self._load_dataset()
        self._prepare_test_dataframe()

    # ----------------------------------------------------
    # DATASET
    # ----------------------------------------------------
    def _load_dataset(self):
        self.df = pd.read_csv(self.dataset_csv)

        print(
            f"Dataset cargado con {len(self.df)} registros "
            f"{self.df['cluster'].value_counts().to_dict()}"
        )

    def _prepare_test_dataframe(self):

        df_exponential = self.df[self.df['cluster'] == 0].sample(
            n=200, random_state=42
        )
        df_stationary = self.df[self.df['cluster'] == 1].sample(
            n=200, random_state=42
        )
        df_lag = self.df[self.df['cluster'] == 2].sample(
            n=200, random_state=42
        )

        self.df_test = pd.concat(
            [df_exponential, df_stationary, df_lag],
            ignore_index=True
        )

        self.df_test.to_csv(self.output_dir / "df_test.csv", index=False)

    # ----------------------------------------------------
    # ROI + FEATURES
    # ----------------------------------------------------
    def recortar_roi(self, imagen, tamano=250, pos_x=0, pos_y=0):
        return imagen[pos_y:pos_y+tamano, pos_x:pos_x+tamano]

    def obtenerRGBI(self, roi_rgb):
        r, g, b = cv2.split(roi_rgb)
        i = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

        return {
            "mean_R": np.mean(r),
            "mean_G": np.mean(g),
            "mean_B": np.mean(b),
            "mean_I": np.mean(i),
            "std_R": np.std(r),
            "std_G": np.std(g),
            "std_B": np.std(b),
            "std_I": np.std(i),
        }

    # ----------------------------------------------------
    # TRANSFORMACIONES
    # ----------------------------------------------------
    def variacion_iluminacion(self, imagen, factor):
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + factor / 100), 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def ruido_gaussiano(self, imagen, media=0, sigma=25):
        ruido = np.random.normal(media, sigma, imagen.shape).astype(np.uint8)
        return cv2.add(imagen, ruido)

    def desplazamiento_roi(self, imagen, tamaño, margenX, margenY):
        alto, ancho = imagen.shape[:2]
        centro_x, centro_y = ancho // 2, alto // 2

        pos_x = max(0, centro_x - tamaño // 2 + margenX)
        pos_y = max(0, centro_y - tamaño // 2 + margenY)

        pos_x = min(pos_x, ancho - tamaño)
        pos_y = min(pos_y, alto - tamaño)

        return self.recortar_roi(imagen, tamaño, pos_x, pos_y)

    def blur(self, imagen, ksize=5):
        return cv2.GaussianBlur(imagen, (ksize, ksize), 0)

    def burbujas(self, imagen, num_burbujas=5, radio_max=20):
        img = imagen.copy()
        alto, ancho = imagen.shape[:2]

        for _ in range(num_burbujas):
            x = random.randint(0, ancho - 1)
            y = random.randint(0, alto - 1)
            radio = random.randint(5, radio_max)

            cv2.circle(img, (x, y), radio, (255, 255, 255), -1)
            img = cv2.addWeighted(img, 0.5, imagen, 0.5, 0)

        return img

    def ruido_sal_pimienta(self, imagen, probabilidad=0.05):
        salida = np.copy(imagen)
        num_pixeles = int(probabilidad * imagen.size)

        for _ in range(num_pixeles):
            x = random.randint(0, imagen.shape[1] - 1)
            y = random.randint(0, imagen.shape[0] - 1)
            salida[y, x] = 255 if random.random() < 0.5 else 0

        return salida

    # ----------------------------------------------------
    # PIPELINE PRINCIPAL
    # ----------------------------------------------------
    def procesar_modificacion(self, nombre_mod):

        df_target = self.df_test.copy()

        print(f"\nProcesando: {nombre_mod}")

        for idx, fila in self.df_test.iterrows():

            img = cv2.imread(str(self.photos_dir / fila['photo_src']))
            if img is None:
                continue

            alto, ancho = img.shape[:2]
            tamano = 220

            if nombre_mod == "desplazamiento_roi":
                roi_mod = self.desplazamiento_roi(img, 250, 0, 0)

            else:
                pos_x = (ancho - tamano)//2
                pos_y = (alto - tamano)//2
                roi = self.recortar_roi(img, tamano, pos_x, pos_y)

                if nombre_mod == "variacion_iluminacion":
                    roi_mod = self.variacion_iluminacion(
                        roi, random.randint(-30, 30)
                    )

                elif nombre_mod == "ruido_gaussiano":
                    roi_mod = self.ruido_gaussiano(
                        roi, sigma=random.choice([5,10,15])
                    )

                elif nombre_mod == "blur":
                    roi_mod = self.blur(roi, random.choice([1,3,5]))

                elif nombre_mod == "burbujas":
                    roi_mod = self.burbujas(roi, 2, 5)

                elif nombre_mod == "ruido_sal_pimienta":
                    roi_mod = self.ruido_sal_pimienta(
                        roi, round(random.uniform(0.001,0.005),4)
                    )

            valores = self.obtenerRGBI(roi_mod)

            for k, v in valores.items():
                df_target.at[idx, k.replace("mean_", "value_")
                                 .replace("std_", "value_") +
                                 ("_desv" if "std" in k else "")] = v

        out = self.output_dir / f"df_{nombre_mod}.csv"
        df_target.to_csv(out, index=False)

        print(f"Guardado -> {out}")

    # ----------------------------------------------------
    def ejecutar_todo(self):

        mods = [
            "variacion_iluminacion",
            "ruido_gaussiano",
            "desplazamiento_roi",
            "blur",
            "burbujas",
            "ruido_sal_pimienta",
        ]

        for m in mods:
            self.procesar_modificacion(m)
