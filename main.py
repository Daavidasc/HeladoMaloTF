import cv2
import numpy as np
import mss

# === Parámetros del juego ===
ANCHO_CELDA = 57
ALTO_CELDA = 57
FILAS = 12
COLUMNAS = 15

# === Cargar templates ===
plantillas = {
    1: [cv2.imread("hielo1.png"),
        cv2.imread("hielo2.png"),
        cv2.imread("hielo3.png")],
    2: [cv2.imread("heladoFront.png")],
    3: [cv2.imread("fruta1.png"),
        cv2.imread("fruta2.png"),
        cv2.imread("fruta3.png")]
}
captura = cv2.imread('fondoFinal.png')

# === Crear matriz vacía ===
matriz = np.zeros((FILAS, COLUMNAS), dtype=np.uint8)

# === Buscar cada objeto en pantalla ===
for valor, lista_templates in plantillas.items():
    for template in lista_templates:
        if template is None:
            continue  # evita error si no cargó bien la imagen

        h, w = template.shape[:2]
        resultado = cv2.matchTemplate(captura, template, cv2.TM_CCOEFF_NORMED)
        umbral = 0.8
        loc = np.where(resultado >= umbral)

        for pt in zip(*loc[::-1]):
            centro_x = pt[0] + w // 2
            centro_y = pt[1] + h // 2
            col = centro_x // ANCHO_CELDA
            fila = centro_y // ALTO_CELDA
            print(f"Detectado tipo {valor} en pixel {pt}, celda ({fila}, {col})")

            if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                matriz[fila, col] = valor


print("Matriz del entorno:")
print(matriz)

