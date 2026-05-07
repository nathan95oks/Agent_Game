# server_mock.py - VERSIÓN CORREGIDA
import asyncio
import json
import websockets
import gymnasium as gym
import numpy as np

async def handle_client(websocket):  # <- Elimina el argumento 'path'
    print("🎮 Cliente manual conectado!")
    
    # Crear el entorno LunarLander
    env = gym.make("LunarLander-v2", render_mode="human")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    try:
        while not done:
            # Enviar estado al cliente
            await websocket.send(json.dumps({
                "type": "state",
                "state": obs.tolist(),
                "reward": total_reward
            }))
            
            # Recibir acción del cliente
            response = await websocket.recv()
            data = json.loads(response)
            action = data.get("action", 0)
            
            # Ejecutar acción en el entorno
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            # Mostrar información en consola
            print(f"Step {step}: Acción={action}, Reward={reward:.2f}, Total={total_reward:.2f}", end="\r")
        
        print(f"\n✅ Episodio terminado! Score final: {total_reward:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        env.close()

async def main():
    print("🚀 Servidor Mock de LunarLander")
    print("📡 Escuchando en ws://127.0.0.1:8000")
    print("🎮 Esperando conexión del cliente manual...")
    
    async with websockets.serve(handle_client, "127.0.0.1", 8000):
        await asyncio.Future()  # Ejecutar para siempre

if __name__ == "__main__":
    asyncio.run(main())