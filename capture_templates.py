import cv2
import numpy as np
import mss
import time
import os

def capture_template(template_name, description):
    """Capture a template screenshot"""
    print(f"\nCapturando template: {template_name}")
    print(f"   Descripción: {description}")
    print("   Presiona 'c' para capturar o 'q' para cancelar...")
    
    with mss.mss() as sct:
        while True:
            screen = np.array(sct.grab(sct.monitors[0]))[:, :, :3]
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Captura de Template", cv2.resize(screen_bgr, (1200, 800)))
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                os.makedirs("template", exist_ok=True)
                
                template_path = f"template/{template_name}.png"
                cv2.imwrite(template_path, screen_bgr)
                print(f"   ✅ Template guardado en: {template_path}")
                break
            elif key == ord('q'):
                print(" Captura cancelada")
                break
    
    cv2.destroyAllWindows()

def main():
    print("Capturador de Templates para Bad Ice Cream")
    print("=" * 50)
    
    templates_to_capture = [
        ("death_screen", "Pantalla de muerte con estadísticas (Total Meltdown)"),
        ("skip_ad_button", "Botón 'Skip Ad' en la esquina inferior izquierda"),
        ("instructions_screen", "Pantalla de instrucciones antes de reiniciar")
    ]
    
    print("Este script te ayudará a capturar las imágenes necesarias para detectar:")
    print("1. La pantalla de muerte")
    print("2. El botón para saltar anuncios")
    print("3. La pantalla de instrucciones")
    print("\nPara cada template:")
    print("- Abre el juego y navega a la pantalla correspondiente")
    print("- Presiona 'c' para capturar cuando esté listo")
    print("- Presiona 'q' para cancelar si no quieres capturar")
    
    input("\nPresiona Enter para comenzar...")
    
    for template_name, description in templates_to_capture:
        capture_template(template_name, description)
        
        if template_name != templates_to_capture[-1][0]: 
            input("\nPresiona Enter para continuar con el siguiente template...")
    
    print("\n¡Captura de templates completada!")
    print("Ahora puedes ejecutar el entrenamiento con manejo automático de pantallas.")

if __name__ == "__main__":
    main() 