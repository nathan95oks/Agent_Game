import numpy as np
import random

class TemplateAgent:
    """
    Tu agente personalizado para el torneo.
    Aquí es donde debes implementar tu lógica de control (Heurística, RL, etc.)
    """
    def __init__(self):
        # Aquí puedes cargar modelos o inicializar variables
        pass

    def predict(self, s):
        """
        Recibe el estado 's' y retorna la acción a tomar.
        
        Estado s (8 valores):
        s[0]: Posición Horizontal
        s[1]: Posición Vertical
        s[2]: Velocidad Horizontal
        s[3]: Velocidad Vertical
        s[4]: Ángulo
        s[5]: Velocidad Angular
        s[6]: Contacto Pata Izquierda (0 o 1)
        s[7]: Contacto Pata Derecha (0 o 1)
        
        Acciones:
        0: No hacer nada
        1: Motor Lateral Izquierdo
        2: Motor Principal
        3: Motor Lateral Derecho
        """
        
        # --- IMPLEMENTA TU LÓGICA AQUÍ ---
        # Por ahora, el agente toma una acción aleatoria (¡Cámbialo!)
        accion = random.randint(0, 3)
        
        return accion, None

def get_agent():
    return TemplateAgent()

if __name__ == "__main__":
    print("Agente cargado correctamente.")
