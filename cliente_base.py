import asyncio
import json
import numpy as np
import websockets
from modelo import get_agent

# --- Configuración ---
TEAM_NAME = "Nutella"  # ¡Cambia esto por el nombre de tu equipo!
SERVER_URL = "ws://127.0.0.1:8000"

async def jugar():
    agent = get_agent()
    uri = f"{SERVER_URL}/ws/client/{TEAM_NAME}"
    
    print(f"Conectando al servidor como: {TEAM_NAME}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Conectado. Iniciando simulación...")
            
            while True:
                # 1. Recibir estado del servidor
                try:
                    mensaje = await websocket.recv()
                    data = json.loads(mensaje)
                except Exception as e:
                    print(f"Conexión cerrada por el servidor: {e}")
                    break

                if data.get("type") == "state":
                    obs = np.array(data["state"])
                    reward_actual = data["reward"]
                    
                    # 2. El modelo elige la acción
                    # Nota: predict devuelve (accion, state_info)
                    accion, _ = agent.predict(obs)
                    
                    # 3. Enviar acción al servidor
                    await websocket.send(json.dumps({
                        "action": int(accion)
                    }))
                    
                    # Opcional: imprimir progreso local
                    # print(f"Acción enviada: {accion} | Score: {reward_actual:.2f}", end="\r")

    except Exception as e:
        print(f"Error de conexión: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(jugar())
    except KeyboardInterrupt:
        print("\nSimulación detenida por el usuario.")
