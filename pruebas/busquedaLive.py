import cv2
import numpy as np
import mss
import threading
import time

# === Parámetros de celda y juego ===
ANCHO_CELDA = 57
ALTO_CELDA = 57
FILAS = 12
COLUMNAS = 15
MONITOR = None  # Se definirá automáticamente
SCALE = 1

# === Templates y umbrales ===
plantillas = {
    1: [cv2.imread("hielo1.png"), cv2.imread("hielo2.png"), cv2.imread("hielo3.png")],
    2: [cv2.imread("heladoFront.png"), cv2.imread("heladoLeft.png"),cv2.imread("heladoRight.png"), cv2.imread("heladoBack.png")],
    3: [cv2.imread("fruta1.png"), cv2.imread("fruta2.png"), cv2.imread("fruta3.png"), cv2.imread("fruta4.png"), cv2.imread("fruta52.png"), cv2.imread("fruta6.png")],
    4: [cv2.imread("maloFront1.png"), cv2.imread("maloFront2.png"), cv2.imread("maloFront3.png"),cv2.imread("maloBack1.png"), cv2.imread("maloBack2.png"), cv2.imread("maloBack2.png") ]
}

umbrales = {
    1: [0.75, 0.78, 0.8],
    2: [0.82, 0.65, 0.65, 0.65],
    3: [0.78, 0.73, 0.76],
    4: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
}

# === Variables compartidas ===
frame_actual = None
lock = threading.Lock()
salir = False

# === Detectar la región del juego usando el póster izquierdo ===
def detectar_area_de_juego():
    global MONITOR
    poster = cv2.imread("poster.png", cv2.IMREAD_GRAYSCALE)
    if poster is None:
        print("❌ Error: No se pudo cargar 'poster_izquierdo.png'")
        exit(1)
    w_poster, h_poster = poster.shape[::-1]

    with mss.mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    resultado = cv2.matchTemplate(screen_gray, poster, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

    umbral = 0.7
    if max_val < umbral:
        print("⚠️ No se encontró el póster con suficiente precisión.")
        exit(1)

    print(f"📍 Póster encontrado en {max_loc} con precisión {max_val:.2f}")

    # El juego está inmediatamente a la derecha del póster
    juego_x = max_loc[0] + w_poster
    juego_y = max_loc[1]
    juego_ancho = 855  # Ajusta si tu juego tiene otro ancho
    juego_alto = h_poster  # Igual al alto del póster

    MONITOR = {
        "top": juego_y,
        "left": juego_x,
        "width": juego_ancho,
        "height": juego_alto
    }

    print(f"🎮 Juego detectado en {MONITOR}")

# === Hilo de captura ===
def capturar_pantalla():
    global frame_actual, salir
    with mss.mss() as sct:
        while not salir:
            frame = np.array(sct.grab(MONITOR))[:, :, :3]
            with lock:
                frame_actual = frame
            time.sleep(0.01)  # ~100 FPS

# === Procesamiento y visualización ===
def procesar_y_mostrar():
    global salir, frame_actual

    while not salir:
        with lock:
            if frame_actual is None:
                continue
            captura = frame_actual.copy()

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
                        color = (0, 255, 0) if valor == 1 else (255, 0, 0) if valor == 2 else (0, 0, 255)
                        cv2.rectangle(captura, pt, (pt[0]+w, pt[1]+h), color, 2)
                        texto = f"({fila},{col})"
                        cv2.putText(captura, texto, (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Juego Detectado", captura)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            salir = True
            break

    cv2.destroyAllWindows()

# === Ejecutar todo ===
if __name__ == "__main__":
    detectar_area_de_juego()

    hilo_captura = threading.Thread(target=capturar_pantalla)
    hilo_captura.start()

    procesar_y_mostrar()
    hilo_captura.join()
