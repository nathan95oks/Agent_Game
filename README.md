# Torneo Lunar Lander - Starter Kit

Bienvenido al torneo de Inteligencia Artificial! Tu misión es programar un agente capaz de aterrizar una nave espacial en la superficie lunar de forma segura y eficiente.

## Requisitos Previos

Asegúrate de tener Python 3.8 o superior instalado. Recomendamos usar un entorno virtual.

### Instalación de Dependencias
Ejecuta el siguiente comando para instalar lo necesario:
```bash
pip install websockets numpy "gymnasium[box2d]" pygame
```

## Cómo empezar

1.  **Configura tu equipo:** Abre `cliente_base.py` y cambia la variable `TEAM_NAME` por el nombre de tu equipo.
2.  **Prueba el control manual:** Ejecuta `python manual_client.py` para entender la física de la nave usando las flechas del teclado.
3.  **Implementa tu IA:** Abre el archivo `modelo.py` y modifica el método `predict` con tu lógica de control.
4.  **Prueba en local:** Antes de conectarte al servidor, usa `python entorno_local.py`. Esto abrirá una ventana donde verás a tu agente en acción y recibirás métricas de desempeño (score, combustible, logros) en la terminal.

## Herramientas Disponibles

*   `manual_client.py`: Úsalo para sentir la gravedad y la inercia de la nave tú mismo.
*   `entorno_local.py`: El campo de entrenamiento para tu IA. Úsalo para ajustar tu modelo sin gastar tus 50 intentos oficiales.
*   `cliente_base.py`: El puente hacia el servidor oficial del torneo. Úsalo cuando estés listo para competir.

Tu agente recibirá un vector de estado `s` con 8 valores flotantes:

| Índice | Descripción | Rango |
| :--- | :--- | :--- |
| 0 | Posición Horizontal (x) | -1.0 a 1.0 |
| 1 | Posición Vertical (y) | -1.0 a 1.0 |
| 2 | Velocidad Horizontal | -Inf a +Inf |
| 3 | Velocidad Vertical | -Inf a +Inf |
| 4 | Ángulo de la nave | -Inf a +Inf |
| 5 | Velocidad Angular | -Inf a +Inf |
| 6 | Contacto Pata Izquierda | 0 o 1 |
| 7 | Contacto Pata Derecha | 0 o 1 |

### Acciones posibles
Tu función debe retornar un número entero del 0 al 3:
*   `0`: No hacer nada.
*   `1`: Encender motor lateral izquierdo.
*   `2`: Encender motor principal (empuje hacia arriba).
*   `3`: Encender motor lateral derecho.

## Sistema de Combustible y Puntaje

*   **Límite de Intentos:** Tienes un máximo de **50 intentos**. Solo se guardará tu mejor puntaje histórico.
*   **Eficiencia:** El uso de motores consume combustible.
    *   Motor Principal: -0.4% por frame.
    *   Motores Laterales: -0.1% por frame.
*   **Puntaje:** Aterrizar con éxito da entre +100 y +140 puntos. El ahorro de combustible y la suavidad del aterrizaje aumentan este valor. Estrellarse penaliza fuertemente el puntaje.

## Protocolo de Comunicación

El servidor se comunica mediante WebSockets. 
*   **Envío:** Cada vez que el servidor envía el estado, tu cliente tiene un máximo de **2 segundos** para responder con una acción, de lo contrario la conexión podría cerrarse por timeout.

---
¡Mucha suerte, comandante! El éxito de la misión depende de tu código.
