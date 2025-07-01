import cv2
import numpy as np
import mss
import os

def debug_game_detection():
    """Debug script to help identify game detection issues"""
    
    poster = cv2.imread("template/poster.png", cv2.IMREAD_GRAYSCALE)
    if poster is None:
        print("No se pudo cargar 'template/poster.png'")
        return
    
    print(f"Poster template size: {poster.shape}")
    
    os.makedirs("debug", exist_ok=True)
    
    with mss.mss() as sct:
        print(f"Monitores disponibles: {len(sct.monitors)}")
        for i, monitor in enumerate(sct.monitors):
            print(f"   Monitor {i}: {monitor}")

        for monitor_idx in range(len(sct.monitors)):
            print(f"\n Probando monitor {monitor_idx}...")
            
            try:
                screen = np.array(sct.grab(sct.monitors[monitor_idx]))[:, :, :3]
                gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                
                print(f" Tamaño de pantalla: {gray.shape}")
                
                cv2.imwrite(f"debug/screen_monitor_{monitor_idx}.png", screen)
                cv2.imwrite(f"debug/gray_monitor_{monitor_idx}.png", gray)
                
                resultado = cv2.matchTemplate(gray, poster, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(resultado)
                
                print(f" Valor máximo de coincidencia: {max_val:.4f}")
                print(f" Ubicación máxima: {max_loc}")
                
                resultado_normalized = ((resultado - resultado.min()) * 255 / (resultado.max() - resultado.min())).astype(np.uint8)
                cv2.imwrite(f"debug/result_map_monitor_{monitor_idx}.png", resultado_normalized)
                
                screen_with_rect = screen.copy()
                cv2.rectangle(screen_with_rect, max_loc, 
                             (max_loc[0] + poster.shape[1], max_loc[1] + poster.shape[0]), 
                             (0, 255, 0), 2)
                cv2.imwrite(f"debug/best_match_monitor_{monitor_idx}.png", screen_with_rect)
                
                if max_val >= 0.3: 
                    print(f"¡Coincidencia encontrada en monitor {monitor_idx}!")
                    juego_x = max_loc[0] + poster.shape[1]
                    juego_y = max_loc[1]
                    MONITOR = {
                        "top": juego_y,
                        "left": juego_x,
                        "width": 855,
                        "height": poster.shape[0]
                    }
                    print(f" ROI calculado: {MONITOR}")
                    return MONITOR
                else:
                    print(f"Coincidencia insuficiente (umbral: 0.3)")
                    
            except Exception as e:
                print(f"Error al procesar monitor {monitor_idx}: {e}")
    
    print("\n No se pudo detectar el área de juego en ningún monitor")
    return None

if __name__ == "__main__":
    debug_game_detection() 