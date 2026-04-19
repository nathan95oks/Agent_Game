import gymnasium as gym
import time
import numpy as np
from modelo import get_agent

def simular_local():
    """
    Ejecuta una simulación local del entorno LunarLander para pruebas.
    """
    print("Iniciando simulador local para pruebas...")
    
    # Inicializar el entorno en modo visual (human)
    try:
        env = gym.make("LunarLander-v3", render_mode="human")
    except Exception as e:
        print(f"Error al cargar el entorno: {e}")
        print("Asegúrate de tener instalado: pip install 'gymnasium[box2d]'")
        return

    agent = get_agent()
    
    intentos = 5
    for i in range(intentos):
        print(f"\n--- Intento de Prueba {i+1} ---")
        obs, _ = env.reset()
        done = False
        score = 0
        fuel = 100.0
        start_time = time.time()

        while not done:
            # Pedir acción al agente
            action, _ = agent.predict(obs)
            
            # Cálculo de combustible (simulando lógica del servidor)
            if action == 2: fuel -= 0.4
            elif action in [1, 3]: fuel -= 0.1
            fuel = max(0, fuel)

            # Paso en el simulador
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

            # Pequeña pausa para que sea visible (aprox 30 FPS)
            time.sleep(0.01)

        duration = time.time() - start_time
        print(f"Resultado: {'ÉXITO' if score > 0 else 'CHOQUE'}")
        print(f"Score Final: {score:.2f}")
        print(f"Combustible restante: {fuel:.1f}%")
        print(f"Duración: {duration:.2f}s")
        
        # Evaluar logros (simulado)
        if score > 0:
            if fuel > 85: print("LOGRO: MASTER PILOT")
            if duration < 15: print("LOGRO: SPEED DEMON")
            if score > 250: print("LOGRO: BULLSEYE")

        time.sleep(1) # Pausa entre intentos

    env.close()
    print("Pruebas locales terminadas.")

if __name__ == "__main__":
    simular_local()
