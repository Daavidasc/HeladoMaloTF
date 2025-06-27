import cv2
import numpy as np
import mss
import time

# === Coordenadas relativas al área del juego detectada ===
roi_offset_x = 178
roi_offset_y = 35
roi_width = 27
roi_height = 30
MONITOR = None

def detectar_area_de_juego():
    global MONITOR
    poster = cv2.imread("poster.png", cv2.IMREAD_GRAYSCALE)
    if poster is None:
        print("❌ No se pudo cargar 'poster.png'")
        exit(1)

    w_poster, h_poster = poster.shape[::-1]
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    resultado = cv2.matchTemplate(screen_gray, poster, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

    if max_val < 0.7:
        print("⚠️ No se encontró el póster con precisión suficiente.")
        exit(1)

    juego_x = max_loc[0] + w_poster
    juego_y = max_loc[1]
    juego_ancho = 855
    juego_alto = h_poster

    MONITOR = {
        "top": juego_y,
        "left": juego_x,
        "width": juego_ancho,
        "height": juego_alto
    }

    print(f"🎮 Juego detectado en: {MONITOR}")

def comparar_puntuacion(prev_roi, curr_roi, umbral=0.99):
    res = cv2.matchTemplate(curr_roi, prev_roi, cv2.TM_CCOEFF_NORMED)
    max_sim = cv2.minMaxLoc(res)[1]
    return max_sim < umbral

def main():
    detectar_area_de_juego()
    roi_anterior = None

    with mss.mss() as sct:
        while True:
            captura = np.array(sct.grab(MONITOR))[:, :, :3]
            roi_actual = captura[roi_offset_y:roi_offset_y + roi_height,
                                 roi_offset_x:roi_offset_x + roi_width]

            # Mostrar el recorte de puntuación en vivo
            cv2.imshow("🟨 Zona de Puntuación", roi_actual)

            if roi_anterior is not None:
                cambio = comparar_puntuacion(roi_anterior, roi_actual)
                print("✅ PUNTUACIÓN CAMBIÓ" if cambio else "⏳ Puntuación igual")
            else:
                print("🔍 Inicializando...")

            roi_anterior = roi_actual.copy()
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
