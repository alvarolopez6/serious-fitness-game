# Serious Fitness Game

Videojuego serio de entrenamiento físico mediante visión por computador.
El proyecto utiliza la cámara del equipo para detectar la pose corporal y
contar flexiones y sentadillas.

## Tecnologías

- Python 3.11 o superior.
- `uv` para gestionar el entorno y las dependencias.
- OpenCV para la captura de vídeo.
- MediaPipe Pose Landmarker para detectar el esqueleto corporal.
- NumPy para calcular ángulos 2D.
- SQLite para guardar las sesiones.

No se utiliza Conda.

## Requisitos

- Python 3.11 o superior.
- `uv` instalado.
- Una cámara compatible con OpenCV.

En macOS, la aplicación de terminal debe tener permiso para utilizar la
cámara. Se puede activar desde **Ajustes del Sistema → Privacidad y
seguridad → Cámara**.

## Ejecución

Desde la raíz del proyecto:

```bash
./run.sh
```

El script realiza automáticamente estos pasos:

1. Sincroniza las dependencias mediante `uv`.
2. Comprueba si existe el modelo de MediaPipe.
3. Descarga el modelo si todavía no está disponible.
4. Inicia la aplicación.

También se puede preparar el entorno manualmente:

```bash
uv sync
uv run python download_model.py
uv run python pose_landmarker.py
```

## Modelo de MediaPipe

El modelo oficial se guarda localmente en:

```text
models/pose_landmarker_full.task
```

`download_model.py` solo lo descarga si no existe. El archivo se guarda en un
archivo temporal durante la descarga para evitar conservar un modelo
incompleto si la conexión se interrumpe.

El modelo está excluido de Git mediante `.gitignore`.

## Modos de juego

Primero se selecciona el ejercicio y el modo desde el menú.

### Reto contrarreloj

Se dispone de un tiempo limitado para hacer tantas repeticiones como sea
posible.

La duración actual se configura en `pose_landmarker.py`:

```python
SESSION_DURATION = 10
```

### Entrenamiento libre

No tiene límite de tiempo ni objetivo de repeticiones. El usuario puede
entrenar libremente y terminar pulsando `Enter`.

### Fijar repeticiones

Permite medir cuánto tarda el usuario en completar un número concreto de
repeticiones. Los objetivos disponibles son:

```text
5, 10, 15 y 20 repeticiones
```

El cronómetro comienza después de la cuenta atrás y se detiene al alcanzar el
objetivo.

Todos los modos incluyen una cuenta atrás inicial de 5 segundos.

## Controles

### Menú

```text
1       Seleccionar flexiones
2       Seleccionar sentadillas
m       Seleccionar reto contrarreloj
l       Seleccionar entrenamiento libre
f       Seleccionar o cambiar el objetivo de repeticiones
s       Mostrar estadísticas del modo seleccionado
Enter   Comenzar la sesión
q       Salir
Esc     Salir
```

Al pulsar `f` se alternan los objetivos de:

```text
5 → 10 → 15 → 20 → 5
```

### Durante una sesión

En el entrenamiento libre, `Enter` termina la sesión. En los demás modos la
sesión termina automáticamente al alcanzar el tiempo o el objetivo.

### Pantalla final y estadísticas

```text
r       Volver al menú
q       Salir
Esc     Salir
```

## Detección de pose y ángulos

MediaPipe devuelve 33 landmarks corporales. El programa solo utiliza un
landmark cuando su visibilidad es superior a `0.6`.

Los ángulos se calculan en 2D mediante `numpy.arctan2`:

- Flexiones: hombro izquierdo – codo izquierdo – muñeca izquierda.
- Sentadillas: cadera izquierda – rodilla izquierda – tobillo izquierdo.

El conteo utiliza una máquina de estados con `UP` y `DOWN`. Una repetición se
cuenta solo después de completar la secuencia:

```text
UP → DOWN → UP
```

La histéresis evita que el ruido de la cámara produzca repeticiones falsas.

## Persistencia y estadísticas

Las sesiones terminadas se guardan automáticamente en:

```text
data/sessions.db
```

Cada sesión contiene:

- Fecha y hora.
- Ejercicio.
- Modo.
- Repeticiones.
- Duración.
- Puntuación.

Las estadísticas se filtran por el modo seleccionado. En el modo de objetivo,
cada cantidad de repeticiones se almacena por separado:

```text
TARGET_5
TARGET_10
TARGET_15
TARGET_20
```

La pantalla de estadísticas muestra:

- Número total de sesiones del modo.
- Repeticiones acumuladas.
- Duración media.
- Puntuación media.
- Las tres sesiones más recientes.

La base de datos está excluida de Git porque contiene el historial local del
usuario.

## Estructura del proyecto

```text
.
├── database.py
├── download_model.py
├── pose_landmarker.py
├── pyproject.toml
├── run.sh
├── README.md
├── models/
│   └── pose_landmarker_full.task
└── data/
    └── sessions.db
```

Los archivos del modelo y de la base de datos se generan localmente y están
excluidos de Git.

## Estado actual

La aplicación permite detectar pose, calcular ángulos, contar flexiones y
sentadillas, jugar en tres modos, puntuar sesiones, guardar resultados en
SQLite y consultar estadísticas filtradas por modo.
