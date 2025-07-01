import gymnasium as gym
import cv2
import numpy as np
import time
import mss
from pynput.keyboard import Key, Controller
from data_collector import DataCollector
from game_flow_manager import GameFlowManager

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
        print("No se pudo cargar 'template/poster.png'")
        exit(1)

    with mss.mss() as sct:
        best_match = 0
        best_monitor = None
        best_location = None
        
        print(f"Buscando juego en {len(sct.monitors)} monitor(es)...")
        
        for monitor_idx in range(len(sct.monitors)):
            try:
                screen = np.array(sct.grab(sct.monitors[monitor_idx]))[:, :, :3]
                gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                resultado = cv2.matchTemplate(gray, poster, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(resultado)
                
                print(f"   Monitor {monitor_idx}: coincidencia = {max_val:.4f}")
                
                if max_val > best_match:
                    best_match = max_val
                    best_monitor = monitor_idx
                    best_location = max_loc
                    
            except Exception as e:
                print(f"   Error en monitor {monitor_idx}: {e}")
                continue

        threshold = 0.3  # Reducido de 0.7
        
        if best_match < threshold or best_location is None:
            print(f"No se detectó el área del juego. Mejor coincidencia: {best_match:.4f} (umbral: {threshold})")
            print("Sugerencias:")
            print("   1. Asegúrate de que el juego esté visible y en primer plano")
            print("   2. Verifica que el template/poster.png coincida con el juego actual")
            print("   3. Intenta ajustar la resolución o escala del juego")
            exit(1)

        print(f"Juego detectado en monitor {best_monitor} con coincidencia {best_match:.4f}")
        
        juego_x = best_location[0] + poster.shape[1]
        juego_y = best_location[1]
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

# === Función para mover en una dirección usando pynput ===
keyboard_controller = Controller()

def mover_direccion(direccion, duracion=0.3):
    teclas = [Key.up, Key.down, Key.left, Key.right]
    tecla = teclas[direccion]
    keyboard_controller.press(tecla)
    time.sleep(duracion)
    keyboard_controller.release(tecla)

class BadIceCreamEnv(gym.Env):
    def __init__(self, collect_data=True, csv_path="training_data.csv"):
        super(BadIceCreamEnv, self).__init__()
        self.ROI_JUEGO = detectar_area_de_juego()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(12, 15), dtype=np.uint8)
        self.roi_anterior = None
        self.pasos = 0
        self.MAX_STEPS = 500
        self.frutas_recogidas = 0
        
        # Data collection
        self.collect_data = collect_data
        if self.collect_data:
            self.data_collector = DataCollector(csv_path)
        self.frutas_recogidas_anterior = 0
        
        # Game flow management
        self.game_flow_manager = GameFlowManager()

    def reset(self, seed=None, options=None):
        # Wait for game to be ready before starting
        if not self.game_flow_manager.is_game_ready():
            print("🔄 Game not ready, waiting for restart...")
            self.game_flow_manager.wait_for_game_restart()
        
        self.ROI_JUEGO = detectar_area_de_juego()
        self.pasos = 0
        self.frutas_recogidas = 0
        self.frutas_recogidas_anterior = 0
        self.roi_anterior = None
        time.sleep(2)

        with mss.mss() as sct:
            frame = np.array(sct.grab(self.ROI_JUEGO))[:, :, :3]
            matriz = generar_matriz(frame)

        self.pos_jugador = self.encontrar_posicion(matriz, 2)
        
        # Reset data collector for new episode
        if self.collect_data:
            self.data_collector.reset_episodio()
            
        return matriz, {}

    def step(self, action):
        mover_direccion(action, duracion=0.3)
        self.pasos += 1

        with mss.mss() as sct:
            captura = np.array(sct.grab(self.ROI_JUEGO))[:, :, :3]
            matriz = generar_matriz(captura)
            roi_actual = obtener_roi(captura)

        muerte = detectar_muerte(captura)
        puntaje_cambio = cambio_visual_puntaje(roi_actual, self.roi_anterior)
        self.roi_anterior = roi_actual.copy()

        reward = 0.0
        frutas_recogidas_este_paso = 0
        
        if puntaje_cambio:
            reward += 1.0
            self.frutas_recogidas += 1
            frutas_recogidas_este_paso = 1
        elif muerte:
            reward -= 1.0
        else:
            reward += 0.01
            nueva_pos = self.encontrar_posicion(matriz, 2)
            reward += self.reward_por_distancia(nueva_pos, matriz)

        terminado = muerte or (self.pasos >= self.MAX_STEPS)
        self.pos_jugador = self.encontrar_posicion(matriz, 2)
        
        # Determina pq termino
        causa_terminacion = ""
        if terminado:
            if muerte:
                causa_terminacion = "muerte"
            elif self.pasos >= self.MAX_STEPS:
                causa_terminacion = "max_steps"
        
        if self.collect_data:
            self.data_collector.registrar_paso(
                matriz=matriz,
                accion=action,
                recompensa=reward,
                pos_jugador=self.pos_jugador,
                terminado=terminado,
                causa_terminacion=causa_terminacion,
                frutas_recogidas=frutas_recogidas_este_paso
            )
        
        # manejar pantalla de muerte 
        if muerte:
            print("💀 Agent died - handling death screen...")
            self.game_flow_manager.wait_for_game_restart()
        
        return matriz, reward, terminado, False, {}

    def render(self):
        with mss.mss() as sct:
            captura = np.array(sct.grab(self.ROI_JUEGO))[:, :, :3]
            cv2.imshow("Juego", cv2.resize(captura, (428, 240)))
            cv2.waitKey(1)

    def encontrar_posicion(self, matriz, valor=2):
        ubicaciones = np.argwhere(matriz == valor)
        return tuple(ubicaciones[0]) if len(ubicaciones) > 0 else (-1, -1)

    def reward_por_distancia(self, pos, matriz):
        if pos == (-1, -1):
            return 0.0
        frutas = np.argwhere(matriz == 3)
        if len(frutas) == 0:
            return 0.0
        distancias = [float(np.linalg.norm(np.array(pos) - np.array(f))) for f in frutas]
        min_dist = min(distancias)
        return max(0.5 - (min_dist / 20.0), 0.0)
    
    def guardar_datos(self):
        """Save collected data to CSV"""
        if self.collect_data:
            self.data_collector.guardar_csv()
    
    def obtener_estadisticas(self):
        """Get training statistics"""
        if self.collect_data:
            return self.data_collector.obtener_estadisticas()
        return {}
