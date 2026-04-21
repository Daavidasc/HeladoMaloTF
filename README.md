# HeladoMaloTF

## Autonomous Agent for 'Bad Ice-Cream' via Reinforcement Learning.

##HeladoMaloTF es un sistema avanzado de automatización de juegos que combina técnicas de visión artificial y aprendizaje por refuerzo. El proyecto captura el estado actual del juego Bad Ice-Cream, lo transforma en una representación matricial estructurada y utiliza un modelo de inteligencia artificial (basado en TensorFlow) para tomar decisiones autónomas y jugar el nivel de forma óptima.
##Pipeline del Proyecto

## El sistema opera bajo un flujo de trabajo de ciclo cerrado:

   Captura y Visión (CV): Utiliza OpenCV para monitorizar la pantalla, filtrar colores y detectar las posiciones de los elementos clave (jugador, enemigos, comida, paredes).

   Mapeo de Estados: Los datos visuales se procesan para generar una matriz de estado (Grid Map), que simplifica el entorno del juego en una estructura numérica comprensible para el agente.

   Agente de Decisión (RL): La matriz se introduce en un modelo de red neuronal (TensorFlow), que aprende a predecir la mejor acción posible (moverse, esquivar, recolectar) para maximizar la recompensa.

## Stack Tecnológico

   Lenguaje: Python 3.10

  Computer Vision: OpenCV (Procesamiento de frames y detección de entidades).

   AI/Deep Learning: TensorFlow (Entrenamiento del modelo).

   Representación de Datos: NumPy (Gestión de la matriz del juego).

   Automatización: PyAutoGUI o librerías de control de periféricos (para ejecutar las acciones).
