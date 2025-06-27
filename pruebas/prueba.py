import keyboard
import time

# === Mover con flechas direccionales ===
def mover_direccion(direccion, duracion=0.3):
    teclas = {
        "arriba": "up",
        "abajo": "down",
        "izquierda": "left",
        "derecha": "right"
    }

    if direccion not in teclas:
        print(f"❌ Dirección inválida: {direccion}")
        return

    print(f"➡️ Moviendo: {direccion}")
    keyboard.press(teclas[direccion])
    time.sleep(duracion)  # Mantiene presionada
    keyboard.release(teclas[direccion])

# === Prueba de movimientos ===
def main():
    time.sleep(2)  # Tiempo para poner foco en el juego
    mover_direccion("arriba")
    mover_direccion("abajo")
    mover_direccion("izquierda")
    mover_direccion("derecha")

if __name__ == "__main__":
    main()
