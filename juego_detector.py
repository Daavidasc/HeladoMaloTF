import cv2
import numpy as np
import mss

# === Parámetros del entorno ===
ANCHO_CELDA = 57
ALTO_CELDA = 57
FILAS = 12
COLUMNAS = 15
ROI_OFFSET = (187, 40, 18, 21)
MONITOR = None

# === Plantillas ===
template_muerte = cv2.imread("template/pantalla_muerte.png", cv2.IMREAD_GRAYSCALE)
UMBRAL_MUERTE = 0.85

plantillas = {
    1: [cv2.imread("template/hielo1.png"), cv2.imread("template/hielo2.png"), cv2.imread("template/hielo3.png")],
    2: [cv2.imread("template/heladoFront.png"), cv2.imread("template/heladoLeft.png"),
        cv2.imread("template/heladoRight.png"), cv2.imread("template/heladoBack.png")],
    3: [cv2.imread("template/fruta1.png"), cv2.imread("template/fruta2.png"), cv2.imread("template/fruta3.png")],
    4: [cv2.imread("template/maloFront1.png"), cv2.imread("template/maloFront2.png"),
        cv2.imread("template/maloFront3.png"), cv2.imread("template/maloBack1.png"),
        cv2.imread("template/maloBack2.png"), cv2.imread("template/maloBack2.png")]
}

umbrales = {
    1: [0.75, 0.78, 0.8],
    2: [0.82, 0.65, 0.65, 0.65],
    3: [0.78, 0.73, 0.76],
    4: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
}

# === Detección del área de juego ===
def detectar_area_de_juego():
    global MONITOR
    poster = cv2.imread("template/poster.png", cv2.IMREAD_GRAYSCALE)
    if poster is None:
        print("❌ No se pudo cargar 'template/poster.png'")
        exit(1)

    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        resultado = cv2.matchTemplate(gray, poster, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

        if max_val < 0.7:
            print("⚠️ No se detectó el área del juego.")
            exit(1)

        juego_x = max_loc[0] + poster.shape[1]
        juego_y = max_loc[1]
        MONITOR = {
            "top": juego_y,
            "left": juego_x,
            "width": 855,
            "height": poster.shape[0]
        }
    return MONITOR

# === Generar matriz del entorno ===
def generar_matriz(captura):
    matriz = np.zeros((FILAS, COLUMNAS), dtype=np.uint8)
    for valor, lista_templates in plantillas.items():
        lista_umbrales = umbrales.get(valor, [0.7] * len(lista_templates))
        for template, umbral in zip(lista_templates, lista_umbrales):
            if template is None:
                continue
            h, w = template.shape[:2]
            resultado = cv2.matchTemplate(captura, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(resultado >= umbral)
            for pt in zip(*loc[::-1]):
                centro_x = pt[0] + w // 2
                centro_y = pt[1] + h // 2
                col = centro_x // ANCHO_CELDA
                fila = centro_y // ALTO_CELDA
                if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                    matriz[fila, col] = valor
    return matriz

# === Detectar muerte ===
def detectar_muerte(frame):
    if template_muerte is None:
        return False
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gris, template_muerte, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= UMBRAL_MUERTE

# === ROI Puntaje ===
def obtener_roi(frame, roi_offset=ROI_OFFSET):
    x, y, w, h = roi_offset
    return frame[y:y + h, x:x + w]

def cambio_visual_puntaje(roi_actual, roi_anterior, umbral=200):
    if roi_anterior is None:
        return False
    diferencia = cv2.absdiff(roi_actual, roi_anterior)
    return np.any(diferencia > umbral)
