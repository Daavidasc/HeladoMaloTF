import pyautogui
import time

print("Presiona Ctrl+C para detener")

try:
    while True:
        x, y = pyautogui.position()
        print(f"\rCoordenadas del mouse: X={x}, Y={y}        ", end='')
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDetenido.")
