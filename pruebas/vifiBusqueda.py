import cv2
import numpy as np
import mss
import time
import keyboard

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
UMBRAL_PLAYER1 = 0.85

# === Plantillas ===
template_muerte = cv2.imread("template/pantalla_muerte.png", cv2.IMREAD_GRAYSCALE)
template_player1 = cv2.imread("template/player1.png", cv2.IMREAD_GRAYSCALE)

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
        cv2.imread("template/heladoRight.png"), cv2.imread("template/heladoBack.png")],

    3: [cv2.imread("template/fruta1.png"), cv2.imread("template/fruta2.png"), cv2.imread("template/fruta3.png"),
        cv2.imread("template/fruta4.png"), cv2.imread("template/fruta5.png"), cv2.imread("template/fruta6.png")],
    4: [cv2.imread("template/maloFront1.png"), cv2.imread("template/maloFront2.png"), cv2.imread("template/maloFront3.png"),
        cv2.imread("template/maloBack1.png"), cv2.imread("template/maloBack2.png"), cv2.imread("template/maloBack2.png"),
        cv2.imread("template/maloLeft1.png"), cv2.imread("template/maloLeft3.png"), cv2.imread("template/maloLeft4.png"), cv2.imread("template/maloLeft5.png"),
        cv2.imread("template/maloRight1.png"), cv2.imread("template/maloRight3.png"), cv2.imread("template/maloRight4.png"), cv2.imread("template/maloRight5.png")]
}

umbrales = {
    1: [0.75, 0.78, 0.8],
    2: [0.82, 0.65, 0.65, 0.65, 0.65],
    3: [0.78, 0.73, 0.76, 0.75, 0.74, 0.73],
    4: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55, 0.55]
}

# === Detectar región del juego ===
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
        print(f"🎮 Juego detectado en: {MONITOR}")

# === Funciones auxiliares ===
def obtener_roi(frame, roi_offset):
    x, y, w, h = roi_offset
    return frame[y:y + h, x:x + w]

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

def detectar_pantalla_de_muerte(frame):
    if template_muerte is None:
        return False
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gris, template_muerte, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= UMBRAL_MUERTE

def detectar_helado_muerto(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for plantilla in template_helado_muerto:
        if plantilla is None:
            continue
        res = cv2.matchTemplate(gris, plantilla, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= UMBRAL_HELADO_MUERTO:
            print(f"🧊 Helado muerto detectado con precisión {max_val:.2f}")
            return True
    return False

def detectar_player1(frame):
    if template_player1 is None:
        return True  # Si no hay imagen, asumimos que no hay anuncios
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gris, template_player1, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= UMBRAL_PLAYER1

# === Bucle principal ===
def main():
    detectar_area_de_juego()
    roi_anterior = None
    helado_ya_murio = False
    en_anuncio = False
    tiempo_ultimo_chequeo_p1 = time.time()

    with mss.mss() as sct:
        while True:
            captura = np.array(sct.grab(MONITOR))[:, :, :3]
            tiempo_actual = time.time()

            # Verificar cada 5 segundos si desapareció el símbolo Player 1 (anuncio)
            if tiempo_actual - tiempo_ultimo_chequeo_p1 >= 5:
                if not detectar_player1(captura):
                    if not en_anuncio:
                        print("📺 Anuncio detectado: símbolo Player 1 desapareció.")
                        en_anuncio = True
                else:
                    if en_anuncio:
                        print("✅ Fin del anuncio. Esperando 2 segundos y reanudando...")
                        time.sleep(2)
                        keyboard.press("space")
                        time.sleep(0.2)
                        keyboard.release("space")
                        en_anuncio = False
                tiempo_ultimo_chequeo_p1 = tiempo_actual

            # Comparar ROI visualmente
            roi_actual = obtener_roi(captura, ROI_OFFSET)
            if roi_anterior is not None:
                diferencia = cv2.absdiff(roi_actual, roi_anterior)
                if np.any(diferencia > 200):
                    print("🔁 ROI cambió")
                else:
                    print("⏳ Sin cambio en ROI")
            else:
                print("🔍 ROI inicializado")
            roi_anterior = roi_actual.copy()

            # Generar matriz del entorno
            matriz = generar_matriz(captura)
            print("🧩 MATRIZ DEL ENTORNO:")
            print(matriz)

            # Detectar muerte del helado
            if not helado_ya_murio and detectar_helado_muerto(captura):
                print("💥 ¡Helado ha muerto! Esperando pantalla de muerte...")
                helado_ya_murio = True

            # Detectar pantalla de muerte y reiniciar
            if helado_ya_murio and detectar_pantalla_de_muerte(captura):
                print("💀 Pantalla de muerte detectada. Presionando [ESPACIO]")
                keyboard.press("space")
                time.sleep(0.2)
                keyboard.release("space")
                helado_ya_murio = False

            print("-" * 40)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
