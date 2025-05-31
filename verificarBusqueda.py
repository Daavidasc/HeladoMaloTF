import cv2
import numpy as np

# Cargar imágenes
original = cv2.imread('fondoFinal.png')
heladoFront = cv2.imread('heladoFront.png')

# Template Matching
heladoFront_match = cv2.matchTemplate(original, heladoFront, cv2.TM_CCOEFF_NORMED)

# Umbral de similitud (ajústalo según sea necesario)
threshold = 0.8  # 1.0 es coincidencia perfecta

# Obtener todas las ubicaciones donde la coincidencia es mayor al umbral
locations = np.where(heladoFront_match >= threshold)
h, w = heladoFront.shape[:2]

# Dibujar rectángulos para todas las coincidencias encontradas
for pt in zip(*locations[::-1]):  # invertir el orden para (x, y)
    cv2.rectangle(original, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# Mostrar resultado
cv2.imshow('Coincidencias encontradas', original)

# Mostrar mapa de similitud (opcional)
match_visual = heladoFront_match.copy()
cv2.normalize(match_visual, match_visual, 0, 255, cv2.NORM_MINMAX)
match_visual = match_visual.astype('uint8')
cv2.imshow('Mapa de coincidencia', match_visual)

cv2.waitKey(0)
cv2.destroyAllWindows()
