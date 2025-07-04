import gymnasium as gym
import numpy as np
import time
import mss
import cv2
import keyboard  # Nuevo módulo para simular teclas

from juego_detector import (
    detectar_area_de_juego,
    generar_matriz,
    detectar_muerte,
    obtener_roi,
    cambio_visual_puntaje,
    esperar_fin_de_anuncio
)

# === Función para mover en una dirección usando `keyboard` ===
def mover_direccion(direccion, duracion=0.3):
    teclas = ["up", "down", "left", "right"]
    tecla = teclas[direccion]

    keyboard.press(tecla)
    time.sleep(duracion)
    keyboard.release(tecla)


class BadIceCreamEnv(gym.Env):
    def __init__(self):
        super(BadIceCreamEnv, self).__init__()
        self.ROI_JUEGO = detectar_area_de_juego()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(13, 15), dtype=np.uint8)
        self.roi_anterior = None
        self.pasos = 0
        self.MAX_STEPS = 500
        self.frutas_recogidas = 0
        esperando_reanudacion = False

    def reset(self, seed=None, options=None):
        self.pasos = 0
        self.frutas_recogidas = 0
        self.roi_anterior = None

        with mss.mss() as sct:
            frame = np.array(sct.grab(self.ROI_JUEGO))[:, :, :3]
            matriz = generar_matriz(frame)

        self.pos_jugador = self.encontrar_posicion(matriz, 2)
        return matriz, {}

    def step(self, action):
        mover_direccion(action, duracion=0.3) 
        self.pasos += 1
        print("Moviéndose hacia:", action," Paso:",self.pasos)
        with mss.mss() as sct:
            captura = np.array(sct.grab(self.ROI_JUEGO))[:, :, :3]
            gris = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)
            matriz = generar_matriz(captura)
            roi_actual = obtener_roi(captura)

        esperar_fin_de_anuncio(gris)

        muerte = detectar_muerte(gris)
        puntaje_cambio = cambio_visual_puntaje(roi_actual, self.roi_anterior)
        self.roi_anterior = roi_actual.copy()

        reward = 0.0
        if puntaje_cambio:
            reward += 1.0
            self.frutas_recogidas += 1
            print('Fruta recogida +1')
        elif muerte:
            reward -= 5.0
            print('Has Muerto -5')
        else:
            reward += 0.03
            nueva_pos = self.encontrar_posicion(matriz, 2)
            reward += self.reward_por_distancia(nueva_pos, matriz)

        terminado = muerte or (self.pasos >= self.MAX_STEPS)
        self.pos_jugador = self.encontrar_posicion(matriz, 2)
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
        distancias = [np.linalg.norm(np.array(pos) - np.array(f)) for f in frutas]
        min_dist = min(distancias)
        return max(0.5 - (min_dist / 20.0), 0.0)