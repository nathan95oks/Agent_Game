import asyncio
import json
import websockets
import pygame
import sys

# --- Configuración ---
TEAM_NAME = "Jugador_Manual"
SERVER_URL = "ws://127.0.0.1:8000"

async def manual_play():
    # Inicializar Pygame para capturar teclado
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Control LunarLander")
    font = pygame.font.SysFont("Arial", 18)

    uri = f"{SERVER_URL}/ws/client/{TEAM_NAME}"
    print(f"Conectando para juego manual como: {TEAM_NAME}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Conectado. ¡Usa las flechas del teclado!")
            
            while True:
                # 1. Recibir estado (aunque no lo usemos para la lógica, el servidor lo envía)
                mensaje = await websocket.recv()
                data = json.loads(mensaje)
                
                if data.get("type") == "state":
                    # 2. Capturar acción del teclado
                    action = 0 # No hacer nada
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:    action = 2 # Motor principal
                    elif keys[pygame.K_LEFT]:  action = 1 # Motor izquierdo
                    elif keys[pygame.K_RIGHT]: action = 3 # Motor derecho
                    
                    # 3. Enviar acción
                    await websocket.send(json.dumps({"action": action}))

                    # Dibujar algo simple en la ventanita de control
                    screen.fill((30, 30, 30))
                    txt = font.render(f"Acción: {action}", True, (255, 255, 255))
                    screen.blit(txt, (50, 80))
                    pygame.display.flip()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(manual_play())
