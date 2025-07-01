from game_flow_manager import GameFlowManager
import time

def test_game_flow():
    print("Probando Game Flow Manager...")
    
    manager = GameFlowManager()
    
    print("\n1. Verificando si el juego está listo...")
    is_ready = manager.is_game_ready()
    print(f"   Juego listo: {is_ready}")
    
    if not is_ready:
        print("\n2. Esperando a que el juego esté listo...")
        success = manager.wait_for_game_restart(max_wait_time=10)
        print(f"   Éxito: {success}")
    
    print("\nPrueba completada!")

if __name__ == "__main__":
    test_game_flow() 