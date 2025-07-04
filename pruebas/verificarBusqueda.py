import cv2
import numpy as np
import mss
import time
import keyboard  # Para presionar teclas automáticamente

# === Parámetros del entorno ===
ANCHO_CELDA = 57
ALTO_CELDA = 57
FILAS = 13
COLUMNAS = 15
ROI_OFFSET = (187, 40, 18, 21)
MONITOR = None

# === Umbrales ===
UMBRAL_MUERTE = 0.85
UMBRAL_HELADO_MUERTO = 0.8
UMBRAL_PLAYER1 = 0.8

# === Plantillas ===
template_muerte = cv2.imread("template/pantalla_muerte.png", cv2.IMREAD_GRAYSCALE)
template_player1 = cv2.imread("template/IconP1.png", cv2.IMREAD_GRAYSCALE)
template_helado_muerto = [
    cv2.imread("template/helado_muerto1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("template/helado_muerto2.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("template/helado_muerto3.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("template/helado_muerto4.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("template/helado_muerto5.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("template/helado_muerto6.png", cv2.IMREAD_GRAYSCALE)
]


plantillas = {
    1: [cv2.imread("template/hielo1.png"), cv2.imread("template/hielo2.png"), cv2.imread("template/hielo3.png")],
    2: [cv2.imread("template/heladoFront.png"), cv2.imread("template/heladoFront1.png"), cv2.imread("template/heladoLeft.png"),
        cv2.imread("template/heladoLeft1.png"), cv2.imread("template/heladoRight.png"), cv2.imread("template/heladoRight1.png"),
        cv2.imread("template/heladoBack.png")],
    3: [cv2.imread("template/fruta1.png"), cv2.imread("template/fruta2.png"), cv2.imread("template/fruta3.png"),
        cv2.imread("template/fruta4.png"), cv2.imread("template/fruta5.png"), cv2.imread("template/fruta6.png")],
    4: [cv2.imread("template/maloFront1.png"), cv2.imread("template/maloFront2.png"), cv2.imread("template/maloFront3.png"),
        cv2.imread("template/maloBack1.png"), cv2.imread("template/maloBack2.png"), cv2.imread("template/maloBack2.png"),
        cv2.imread("template/maloLeft1.png"), cv2.imread("template/maloLeft3.png"), cv2.imread("template/maloLeft4.png"), cv2.imread("template/maloLeft5.png"),
        cv2.imread("template/maloRight1.png"), cv2.imread("template/maloRight3.png"), cv2.imread("template/maloRight4.png"), cv2.imread("template/maloRight5.png")]
}

umbrales = {
    1: [0.75, 0.78, 0.8],
    2: [0.82, 0.65, 0.65, 0.65, 0.65,0.65, 0.65],
    3: [0.78, 0.73, 0.76, 0.75, 0.74, 0.73],
    4: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55, 0.55]
}


# === Funciones auxiliares ===
def detectar_match(frame_gray, template, umbral):
    if template is None:
        return False
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= umbral

def detectar_helado_muerto(frame_gray):
    for template in template_helado_muerto:
        if template is None:
            continue
        if detectar_match(frame_gray, template, UMBRAL_HELADO_MUERTO):
            print("🧊 Helado muerto detectado.")
            return True
    return False

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

def detectar_area_de_juego():
    global MONITOR
    poster = cv2.imread("template/poster.png", cv2.IMREAD_GRAYSCALE)
    if poster is None:
        print("❌ No se pudo cargar 'template/poster.png'")
        exit(1)

    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, poster, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

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
        print(f"🎮 Juego detectado en: {MONITOR}")

def obtener_roi(frame, roi_offset):
    x, y, w, h = roi_offset
    return frame[y:y + h, x:x + w]

# === Bucle principal ===
def main():
    detectar_area_de_juego()
    roi_anterior = None
    helado_ya_murio = False
    esperando_reanudacion = False
    ultimo_chequeo_anuncio = 0

    with mss.mss() as sct:
        while True:
            captura = np.array(sct.grab(MONITOR))[:, :, :3]
            captura_gray = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)

            # Verificación de anuncios cada 5 segundos
            tiempo_actual = time.time()
            if tiempo_actual - ultimo_chequeo_anuncio >= 5:
                hay_player1 = detectar_match(captura_gray, template_player1, UMBRAL_PLAYER1)
                if not hay_player1:
                    print("📺 Anuncio detectado. Pausando acciones...")
                    esperando_reanudacion = True
                elif esperando_reanudacion:
                    print("✅ Anuncio terminó. Reanudando juego...")
                    time.sleep(2)
                    keyboard.press("space")
                    time.sleep(0.3)
                    keyboard.release("space")
                    esperando_reanudacion = False
                ultimo_chequeo_anuncio = tiempo_actual

            if esperando_reanudacion:
                time.sleep(0.1)
                continue

            # Si el helado murió, detener todo hasta pantalla de muerte
            if not helado_ya_murio and detectar_helado_muerto(captura_gray):
                print("💥 ¡Helado ha muerto! Esperando pantalla de muerte...")
                helado_ya_murio = True

            if helado_ya_murio:
                if detectar_match(captura_gray, template_muerte, UMBRAL_MUERTE):
                    print("💀 Pantalla de muerte detectada. Reiniciando...")
                    keyboard.press("space")
                    time.sleep(0.3)
                    keyboard.release("space")
                    print("⏳ Esperando 2 segundos para empezar...")
                    time.sleep(2)
                    keyboard.press("space")
                    time.sleep(0.3)
                    keyboard.release("space")
                    helado_ya_murio = False
                else:
                    print("⌛ Esperando pantalla de muerte...")
                time.sleep(0.5)
                continue

            # Comparar ROI visualmente
            roi_actual = obtener_roi(captura, ROI_OFFSET)
            if roi_anterior is not None:
                diferencia = cv2.absdiff(roi_actual, roi_anterior)
                if np.any(diferencia > 200):
                    print("🔁 ROI cambió")
            else:
                print("🔍 ROI inicializado")
            roi_anterior = roi_actual.copy()

            # Generar matriz
            matriz = generar_matriz(captura)
            print("🧩 MATRIZ DEL ENTORNO:")
            print(matriz)

            print("-" * 50)
            time.sleep(0.2)

if __name__ == "__main__":
    main()
