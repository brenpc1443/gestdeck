# GestDeck - Deep Learning Project

**GestDeck** es un proyecto académico de Deep Learning enfocado en la clasificación de secuencias temporales de gestos de manos mediante visión computacional y redes neuronales profundas (LSTM).

El objetivo principal de este proyecto es analizar datos espaciales extraídos de secuencias de video (landmarks de las manos) para entrenar un modelo profundo que pueda clasificar gestos con alta precisión y exhaustividad.

---

## 🔬 Arquitectura del Modelo (Deep Learning)

El proyecto utiliza **TensorFlow/Keras** para construir la arquitectura de red neuronal y **MediaPipe** como feature extractor pre-entrenado.

1. **Feature Extraction (MediaPipe)**: Reduce la dimensionalidad del problema extrayendo 21 landmarks (x,y) por frame, volviendo al modelo invariante frente al color de piel, iluminación y fondo.
2. **Deep Sequence Modeling (LSTM)**: Dado que los gestos ocurren a lo largo del tiempo, la entrada de forma $(Frames, Features)$ es alimentada a múltiples capas Long Short-Term Memory (LSTM) con Dropout para prevenir el sobreajuste.
3. **Clasificación Probabilística**: Una capa Dense conectada a una función de activación Softmax entrega la probabilidad para cada una de las 9 clases.

---

## 📂 Estructura del Repositorio

El proyecto mantiene una estructura limpia orientada a la ciencia de datos y experimentación:

```
GestDeck/
├── training/                       # Entorno Académico de Deep Learning
│   ├── dataset/                    # Videos crudos y arrays procesados (.npy)
│   ├── models/                     # Modelos de Redes Neuronales guardados (.keras)
│   └── notebooks/                  # Notebooks principales del proyecto
│       ├── 01_Dataset_EDA_y_Pipeline.ipynb            (Extracción, EDA, Limpieza y Aumentación)
│       └── 02_LSTM_Modeling_and_Evaluation.ipynb      (Construcción LSTM, entrenamiento y métricas)
├── requirements.txt                # Dependencias estrictas (TensorFlow, OpenCV, etc)
└── README.md                       # Documentación del proyecto
```

---

## 🚀 Cómo ejecutar este avance

### 1. Preparación del Entorno
Se recomienda utilizar un entorno virtual con **Python 3.11 - 3.13**.

```bash
# Crear el entorno virtual
python -m venv prodeep

# Activar el entorno (Windows PowerShell)
.\prodeep\Scripts\activate

# Instalar las librerías necesarias y el kernel para Jupyter
python -m pip install -r requirements.txt ipykernel
```

### 2. Organización del Dataset
Para que los Notebooks funcionen correctamente, los videos grabados (50 por clase) deben depositarse respetando la siguiente jerarquía:
```
training/dataset/raw_videos/
    ├── PUNO_CERRADO/
    ├── PALMA_ABIERTA/
    ├── INDICE/
    ├── PELLIZCO/
    ├── PELLIZCO_INV/
    ├── EMPUJE/
    ├── SIGUIENTE/
    ├── ANTERIOR/
    └── PAUSA/
```

### 3. Ejecución de Notebooks
Abre los notebooks dentro de `training/notebooks/` usando VS Code o JupyterLab. 
Asegúrate de seleccionar el kernel **prodeep** (`Python Environments -> prodeep`) en la esquina superior derecha y ejecuta las celdas en el orden establecido para replicar el pipeline completo de Deep Learning.

---

## 📊 Clases a Clasificar
El modelo ha sido diseñado para aprender un total de **9 clases** distintas, basadas en la interacción natural que una persona realizaría en una presentación:

1. `PUNO_CERRADO`: Gesto utilizado para achicar (zoom out) un elemento en la presentación.
2. `PALMA_ABIERTA`: Gesto utilizado para soltar o deseleccionar un objeto activo.
3. `INDICE`: Gesto utilizado para apuntar y seleccionar un objeto cercano.
4. `PELLIZCO`: Gesto utilizado para agarrar y arrastrar un objeto por la pantalla.
5. `PELLIZCO_INV`: Gesto (movimiento de pinza hacia afuera) para agrandar (zoom in) un objeto.
6. `EMPUJE`: Gesto para devolver un objeto modificado a su estado y posición original.
7. `SIGUIENTE`: Gesto direccional para avanzar a la siguiente diapositiva.
8. `ANTERIOR`: Gesto direccional para retroceder a la diapositiva anterior.
9. `PAUSA`: Gesto para pausar o reanudar temporalmente el seguimiento de la aplicación.
