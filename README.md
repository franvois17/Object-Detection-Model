# Detector de Inventario

Aplicacion de escritorio para reconocimiento de productos de inventario usando deep learning. Graba un video de cada producto y el sistema lo reconoce en tiempo real con la camara.

## Arquitectura

| Componente | Modelo | Funcion |
|-----------|--------|---------|
| Deteccion | YOLOv8-nano | Detecta objetos en frames y camara en vivo |
| Segmentacion | GrabCut (OpenCV) | Remueve el fondo de los recortes |
| Reconocimiento | EfficientNet-V2-S | Extrae embeddings de 1280 dimensiones |
| Busqueda | FAISS (KNN) | Clasifica por similitud coseno |
| Clasificador | Linear head (fine-tuned) | Clasificacion precisa con entrenamiento completo |

## Flujo de uso

1. **Subir video** - Graba un video de 15-60 segundos del producto desde varios angulos
2. **Procesamiento** - El sistema extrae frames, detecta el objeto, remueve el fondo y guarda recortes limpios
3. **Multiples videos** - Puedes agregar mas videos del mismo producto para mejorar el reconocimiento
4. **Entrenamiento** - Modo incremental (segundos, solo FAISS) o completo (minutos, fine-tune + FAISS)
5. **Reconocimiento en vivo** - Abre la camara y el sistema identifica productos en tiempo real

## Requisitos

- Python 3.10+
- macOS (MPS) o Windows (CUDA) o CPU

## Instalacion

```bash
# Clonar repositorio
git clone https://github.com/franvois17/Object-Detection-Model.git
cd Object-Detection-Model

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecucion

```bash
python -m src.app.main
```

## Stack tecnologico

- **UI**: PySide6 (tema oscuro Catppuccin)
- **Deep Learning**: PyTorch, torchvision
- **Deteccion**: Ultralytics YOLOv8
- **Vision**: OpenCV (GrabCut, captura de camara)
- **Embeddings**: FAISS (busqueda por similitud)
- **Augmentacion**: Albumentations (rotaciones, color jitter, reemplazo de fondo aleatorio)
- **Base de datos**: SQLite con SQLAlchemy
- **Dispositivo**: Deteccion automatica CUDA / MPS / CPU

## Estructura del proyecto

```
src/
├── app/                    # UI PySide6
│   ├── main.py             # Entry point
│   ├── main_window.py      # Ventana principal con sidebar
│   ├── styles.py           # Tema oscuro
│   ├── pages/              # Paginas de la app
│   └── widgets/            # Widgets reutilizables
├── core/                   # Configuracion y device
├── video/                  # Pipeline de video
│   ├── frame_extractor.py  # Video -> frames
│   ├── object_detector.py  # YOLOv8 deteccion
│   ├── object_segmenter.py # SAM2 segmentacion (opcional)
│   └── cropper.py          # Recorte + GrabCut
├── ml/                     # Machine learning
│   ├── backbone.py         # EfficientNet-V2-S
│   ├── embeddings.py       # FAISS index
│   ├── classifier.py       # Cabeza clasificadora
│   ├── inference.py        # Inferencia unificada
│   └── augmentations.py    # Data augmentation
├── data/                   # Base de datos y datasets
└── workers/                # QThreads para procesamiento
```

## Mejoras de reconocimiento

- **Remocion de fondo**: GrabCut elimina el fondo de cada recorte, dejando solo el objeto sobre fondo blanco
- **Augmentacion de fondo**: Durante entrenamiento, el fondo blanco se reemplaza aleatoriamente por colores solidos para que el modelo no memorice el fondo
- **Multiples videos**: Se pueden agregar varios videos del mismo producto desde diferentes angulos y fondos
- **Suavizado temporal**: Ventana deslizante de 5 frames con voto mayoritario para estabilizar predicciones en camara
