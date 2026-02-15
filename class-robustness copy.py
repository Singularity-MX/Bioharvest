import pandas as pd
import numpy as np
import random
import cv2
import os

from pathlib import Path

# Ruta absoluta del script
BASE_DIR = Path(__file__).resolve().parent

# Subir hasta la raíz del proyecto (Bioharvest)
PROJECT_ROOT = BASE_DIR.parent.parent

### Configuración inicial
path_photos = PROJECT_ROOT / 'data/'
data_original = PROJECT_ROOT / 'data/database/labeled-dataset.csv'
output_folder = PROJECT_ROOT / 'test/Robustness-tests/generated-datasets/'

# Crear la carpeta de resultados si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## Cargar data_original
df = pd.read_csv(data_original)
#imprimir cuantos registros tiene el df de cada cluster
print(f"Data original cargado con {len(df)} registros, con {df['cluster'].value_counts().to_dict()} registros por cluster.")
# Tomar 150 registros de cada cluster (50 de cada uno para 3 modificaciones)
df_exponential = df[df['cluster'] == 0].sample(n=200, random_state=42)
df_stationary = df[df['cluster'] == 1].sample(n=200, random_state=42)
df_lag = df[df['cluster'] == 2].sample(n=200, random_state=42)

# Concatenar los dataframes
df_test = pd.concat([df_exponential, df_stationary, df_lag], ignore_index=True)
#guardar como csv el test
df_test.to_csv(output_folder / 'df_test.csv', index=False)


print(f"DataFrame de prueba creado con {len(df_test)} registros., con {df_test['cluster'].value_counts().to_dict()} registros por cluster.")

# ----------------------------------------------------------------- CREAR LOS DF PARA VARIAR LOS DATOS
df_variacion_iluminacion = df_test.copy()
df_ruido_gaussiano = df_test.copy()
df_desplazamiento_roi = df_test.copy()
df_blur = df_test.copy()
df_burbujas = df_test.copy()
df_ruido_sal_pimienta = df_test.copy()

# ----------------------------------------------------------------- FUNCIONES DE PROCESAMIENTO  
def recortar_roi(imagen, tamano=250, pos_x=0, pos_y=0):
    """Recorta una región cuadrada (ROI) de una imagen dada."""
    return imagen[pos_y:pos_y+tamano, pos_x:pos_x+tamano]

def obtenerRGBI(roi_rgb):
    """Calcula los promedios y desviaciones estándar de los canales RGB e intensidad (I)."""
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

#/////////////////////////////////// METODOS DE VARIACION ////////////////////////////////////////
def variacion_iluminacion(imagen, factor):   
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + factor / 100), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def ruido_gaussiano(imagen, media=0, sigma=25):
    ruido = np.random.normal(media, sigma, imagen.shape).astype(np.uint8)
    return cv2.add(imagen, ruido)

def desplazamiento_roi(imagen, tamaño, margenX, margenY):
    alto, ancho = imagen.shape[:2]
    centro_x, centro_y = ancho // 2, alto // 2
    pos_x = max(0, centro_x - tamaño // 2 + margenX)
    pos_y = max(0, centro_y - tamaño // 2 + margenY)
    pos_x = min(pos_x, ancho - tamaño)
    pos_y = min(pos_y, alto - tamaño)
    return recortar_roi(imagen, tamano=tamaño, pos_x=pos_x, pos_y=pos_y)

def blur(imagen, ksize=5):
    return cv2.GaussianBlur(imagen, (ksize, ksize), 0)

def burbujas(imagen, num_burbujas=5, radio_max=20):
    imagen_burbujas = imagen.copy()
    alto, ancho = imagen.shape[:2]
    
    for _ in range(num_burbujas):
        x = random.randint(0, ancho - 1)
        y = random.randint(0, alto - 1)
        radio = random.randint(5, radio_max)
        alpha = 0.5
        cv2.circle(imagen_burbujas, (x, y), radio, (255, 255, 255), -1)
        imagen_burbujas = cv2.addWeighted(imagen_burbujas, 1 - alpha, imagen, alpha, 0)
    
    return imagen_burbujas

def ruido_sal_pimienta(imagen, probabilidad=0.05):
    salida = np.copy(imagen)
    num_pixeles = int(probabilidad * imagen.size)
    
    for _ in range(num_pixeles):
        x = random.randint(0, imagen.shape[1] - 1)
        y = random.randint(0, imagen.shape[0] - 1)
        if random.random() < 0.5:
            salida[y, x] = 255  # Sal
        else:
            salida[y, x] = 0    # Pimienta
            
    return salida

# ---------------------------------------------------------------------------------------- MAIN
def procesar_modificacion(df_target, modificacion, params):
    print(f"\nProcesando modificación: {modificacion}")
    
    for idx, fila in df_test.iterrows():
        # Cargar la imagen original
        imagen_original = cv2.imread(str(path_photos / fila['photo_src']))

        if imagen_original is None:
            print(f"Error al cargar imagen: {path_photos / fila['photo_src']}")

            continue
            
        alto, ancho = imagen_original.shape[:2]
        tamano = 220
        
        try:
            if modificacion == "desplazamiento_roi":
                # Para desplazamiento usamos la imagen completa
                roi_mod = desplazamiento_roi(
                    imagen_original, 
                    tamaño=250,
                    margenX=0,#random.randint(-50, 50),
                    margenY=0#random.randint(-50, 50)
                )
            else:
                # Para otras modificaciones recortamos primero el ROI central
                pos_x = (ancho - tamano) // 2
                pos_y = (alto - tamano) // 2
                roi = recortar_roi(imagen_original, tamano=tamano, pos_x=pos_x, pos_y=pos_y)
                
                if modificacion == "variacion_iluminacion":
                    roi_mod = variacion_iluminacion(roi, random.randint(-30, 30))
                elif modificacion == "ruido_gaussiano":
                    roi_mod = ruido_gaussiano(roi, media=0, sigma=random.choice([5, 10, 15]))
                elif modificacion == "blur":
                    roi_mod = blur(roi, ksize=random.choice([1, 3, 5])) #3, 5, 7
                elif modificacion == "burbujas":
                    roi_mod = burbujas(roi, num_burbujas=random.randint(1, 2), radio_max=5)
                elif modificacion == "ruido_sal_pimienta":
                    roi_mod = ruido_sal_pimienta(roi, probabilidad=round(random.uniform(0.001, 0.005), 4))
            
            # Obtener valores RGBI de la ROI modificada
            valores = obtenerRGBI(roi_mod)
            
            # Actualizar el dataframe target
            df_target.at[idx, 'value_R'] = valores['mean_R']
            df_target.at[idx, 'value_G'] = valores['mean_G']
            df_target.at[idx, 'value_B'] = valores['mean_B']
            df_target.at[idx, 'value_I'] = valores['mean_I']
            df_target.at[idx, 'value_R_desv'] = valores['std_R']
            df_target.at[idx, 'value_G_desv'] = valores['std_G']
            df_target.at[idx, 'value_B_desv'] = valores['std_B']
            df_target.at[idx, 'value_I_desv'] = valores['std_I']
            
        except Exception as e:
            print(f"Error procesando imagen {fila['photo_src']}: {str(e)}")
    
    # Guardar el dataframe modificado
    nombre_archivo = f"df_{modificacion}.csv"
    df_target.to_csv(output_folder / nombre_archivo, index=False)

    print(f"Archivo guardado: {nombre_archivo}")

# Procesar todas las modificaciones
modificaciones = [
    ("variacion_iluminacion", df_variacion_iluminacion),
    ("ruido_gaussiano", df_ruido_gaussiano),
    ("desplazamiento_roi", df_desplazamiento_roi),
    ("blur", df_blur),
    ("burbujas", df_burbujas),
    ("ruido_sal_pimienta", df_ruido_sal_pimienta)
]

for nombre, df_mod in modificaciones:
    procesar_modificacion(df_mod, nombre, None)

print("\nProceso completado. Todos los archivos han sido guardados en:", output_folder)

# ---------------------------------------------------------------------------------------- 
# COMPARACIÓN DE UNA MUESTRA ALEATORIA
print("\n" + "="*80)
print(" COMPARACIÓN DE VALORES ORIGINALES VS MODIFICADOS ".center(80, '='))
print("="*80)

# 1. Seleccionar una fila aleatoria del df_test
fila_original = df_test.sample(n=1).iloc[0]
id_muestra = fila_original['id']
print(f"\nMuestra seleccionada (ID: {id_muestra}):")
print(f"Imagen: {fila_original['photo_src']} | Cluster: {fila_original['cluster']}")
print("\nValores ORIGINALES (df_test):")
print(f"R: {fila_original['value_R']:.2f} ± {fila_original['value_R_desv']:.2f}")
print(f"G: {fila_original['value_G']:.2f} ± {fila_original['value_G_desv']:.2f}")
print(f"B: {fila_original['value_B']:.2f} ± {fila_original['value_B_desv']:.2f}")
print(f"I: {fila_original['value_I']:.2f} ± {fila_original['value_I_desv']:.2f}")

# 2. Buscar la misma fila en cada DataFrame modificado
modificaciones = {
    "Variación Iluminación": df_variacion_iluminacion,
    "Ruido Gaussiano": df_ruido_gaussiano,
    "Desplazamiento ROI": df_desplazamiento_roi,
    "Blur": df_blur,
    "Burbujas": df_burbujas,
    "Ruido Sal/Pimienta": df_ruido_sal_pimienta
}

for nombre, df_mod in modificaciones.items():
    fila_mod = df_mod[df_mod['id'] == id_muestra].iloc[0]
    print(f"\nValores MODIFICADOS ({nombre}):")
    print(f"R: {fila_mod['value_R']:.2f} ± {fila_mod['value_R_desv']:.2f} (ΔR: {fila_mod['value_R'] - fila_original['value_R']:+.2f})")
    print(f"G: {fila_mod['value_G']:.2f} ± {fila_mod['value_G_desv']:.2f} (ΔG: {fila_mod['value_G'] - fila_original['value_G']:+.2f})")
    print(f"B: {fila_mod['value_B']:.2f} ± {fila_mod['value_B_desv']:.2f} (ΔB: {fila_mod['value_B'] - fila_original['value_B']:+.2f})")
    print(f"I: {fila_mod['value_I']:.2f} ± {fila_mod['value_I_desv']:.2f} (ΔI: {fila_mod['value_I'] - fila_original['value_I']:+.2f})")

print("\n" + "="*80)