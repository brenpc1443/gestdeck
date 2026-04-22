# GestDeck — Modo Virtual (Ghost Mode)

Aplicación de escritorio que permite manipular objetos digitales sobre diapositivas mediante **gestos naturales de las manos**, sin proyector y sin marcadores ArUco. Diseñada para exposiciones por Zoom/Meet/Teams o para un monitor secundario.

> **Modo Fantasma:** solo se ven las diapositivas y los objetos que reaccionan a tus gestos. El expositor no aparece en la salida. La superposición de tu imagen (si la quieres) se la dejas a Zoom/Meet como PIP.

Proyecto académico de la clase de **Deep Learning 2026**.

---

## Arquitectura

```
┌─────────────────────────────────┐      ┌───────────────────────────────┐
│  Backend Python (core/)         │      │  Frontend Electron + React    │
│  - MediaPipe Tasks (Holistic)   │  WS  │  - Editor de objetos          │
│  - LSTM dual (dom + apoyo)      │◄────►│  - Ventana Presentador FS     │
│  - Motor de física de objetos   │ ~1ms │  - Ventana de Control         │
│  - Mapeo virtual cámara→slide   │      │  - Módulo de entrenamiento    │
└─────────────────────────────────┘      └───────────────────────────────┘
```

**Clasificación dual por mano:**

| Mano | Rol | Clases entrenables |
|---|---|---|
| Dominante | Manipula objetos | PUNO_CERRADO, PALMA_ABIERTA, PALMA_VERTICAL, INDICE, PELLIZCO, PELLIZCO_INV, EMPUJE |
| Apoyo | Navega y pausa | SIGUIENTE, ANTERIOR, PAUSA |

Cada mano tiene su propio LSTM (`gestures_model_dominant.h5` / `gestures_model_support.h5`) entrenado con landmarks normalizados (muñeca absoluta + 20 landmarks relativos escalados).

---

## Requisitos

- **Python 3.10–3.13** (probado en 3.13)
- **Node.js 18+**
- **Webcam**
- Windows 10/11 (probado), macOS y Linux deberían funcionar
- ~1.5 GB libres (TensorFlow + MediaPipe + node_modules)

---

## Instalación

### 1. Clonar

```bash
git clone <url-del-repo> GestDeck
cd GestDeck
```

### 2. Backend (Python)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> En el primer arranque del backend se descargará automáticamente el modelo de MediaPipe Holistic (`holistic_landmarker.task`, ~50 MB) a `core/models/`. No requiere acción manual.

### 3. Frontend (Electron + React)

```bash
cd app
npm install
cd ..
```

---

## Uso

### Arranque rápido

**Windows:**

```powershell
.\start.bat
```

**macOS / Linux:**

```bash
./start.sh
```

Ambos scripts activan el venv, lanzan el backend Python y el frontend Electron con Vite.

Con una sesión guardada:

```powershell
.\start.bat sessions\ejemplo_demo\config.json
```

### Flujo de trabajo

1. **Abre el Editor** (ventana de Control) y carga tu archivo de diapositivas.
2. **Entrena tus gestos** desde la pantalla **🎯 Entrenamiento**:
   - Elige **Dominante** o **Apoyo**.
   - Por cada clase: ● **Grabar nueva** (barra de 1 s capturando 30 frames). Repite ~15–20 muestras por clase.
   - Pulsa **Reentrenar** — se entrena la mano seleccionada con *early stopping* + *data augmentation ×3*. Val acc típica ~94 %.
3. **Iniciar presentación** — abre la Ventana Presentador fullscreen.
4. **Compartir por Zoom / Meet / Teams**: *Share Screen → Window → GestDeck Presentador*.
   - (Opcional) Mantén tu cámara activa en la plataforma para aparecer como PIP.

### Gestos cableados (tras entrenar)

| Mano | Gesto | Acción |
|---|---|---|
| Dominante | INDICE | Seleccionar objeto cercano (única vía de selección) |
| Dominante | PELLIZCO | Arrastrar objeto activo |
| Dominante | PALMA_ABIERTA | Soltar / deseleccionar |
| Dominante | PALMA_VERTICAL | Mostrar / flotar |
| Dominante | PELLIZCO_INV | Agrandar (zoom in) |
| Dominante | PUNO_CERRADO | Achicar (zoom out) |
| Dominante | EMPUJE | Devolver al origen |
| Apoyo | SIGUIENTE / ANTERIOR | Cambiar de slide (cooldown 1.2 s) |
| Apoyo | PAUSA | Toggle pausa (1ª aparición pausa, 2ª reanuda) |

La deselección es automática si la mano se aleja > 2.5 × `grab_radius`.

---

## Estructura de carpetas

```
GestDeck/
├── core/                           # Backend Python
│   ├── main.py                     # Orquestador + perception loop
│   ├── camera.py                   # OpenCV + CLAHE
│   ├── mediapipe_tracker.py        # MediaPipe Tasks API (Holistic)
│   ├── virtual_mapper.py           # Mapeo cámara → slide
│   ├── gesture_classifier.py       # LSTM + fallback rule-based
│   ├── training_manager.py         # Grabación + entrenamiento in-app
│   ├── object_engine.py            # Física y estado de objetos
│   ├── slide_reader.py             # Lee .pptx
│   ├── websocket_server.py
│   └── models/                     # Pesos (auto-descargados/entrenados)
├── training/
│   ├── record_gestures.py          # CLI legacy (no recomendado)
│   ├── train_lstm.py               # CLI legacy
│   └── dataset/                    # Dataset del flujo CLI legacy
├── app/                            # Electron + React + Vite + Tailwind
│   ├── main.js
│   ├── preload.js
│   └── src/
│       ├── App.jsx
│       ├── components/
│       ├── hooks/
│       ├── windows/                # Editor, Presenter, Training
│       └── styles/
├── assets/
│   ├── objects/                    # SVGs de ejemplo
│   └── samples/
├── sessions/
│   └── ejemplo_demo/
│       └── config.json
├── docs/                           # Documentación técnica
├── start.bat | start.sh
└── requirements.txt
```

---

## Entrenamiento y versionado de modelos

- Los datasets se guardan en `training/dataset_inapp_v{N}/<mano>/<CLASE>/*.npy` donde `N` = `MODEL_ARCH_VERSION` (actualmente **4**).
- Cada modelo lleva un archivo `.version` junto al `.h5`. Al arrancar, el clasificador descarta modelos cuya versión no coincida con la arquitectura actual.
- Si modificas features (`HandLandmarks.flat_xy`) o la arquitectura del LSTM, **sube `MODEL_ARCH_VERSION`** en `core/training_manager.py` para invalidar automáticamente los artefactos viejos.

**Arquitectura LSTM:**
```
LSTM(64, return_sequences=True) → Dropout(0.4)
  → LSTM(32) → Dropout(0.4)
  → Dense(32, relu) → Dropout(0.5)
  → Dense(K, softmax)
```

**Optimizaciones aplicadas:**
- Split estratificado 85/15 por clase
- Data augmentation ×3 (jitter σ=0.01, escalado ±5 %, shift temporal ±2 frames)
- Early stopping `patience=8`, `restore_best_weights=True`
- Inferencia con `tf.function` + `input_signature` fija + warmup
- Fallback rule-based solo en la mano dominante (rescata si LSTM < 0.35)

---

## Protocolo WebSocket (resumen)

Ver `docs/websocket_protocol.md` para el detalle. Ejemplo de frame:

```json
{
  "tipo": "frame",
  "gesto": "PELLIZCO",
  "gesto_apoyo": "SIGUIENTE",
  "confianza": 0.94,
  "mano": { "x": 0.45, "y": 0.62 },
  "objeto_activo": "cpu_icon",
  "estado_objeto": "EN_MANO",
  "slide_actual": 3,
  "paused": false
}
```

---

## Notas técnicas

- **MediaPipe:** se usa la **Tasks API** (`HolisticLandmarker`, `RunningMode.VIDEO`). El API legacy `mp.solutions.holistic` fue removido en `mediapipe 0.10.33` (la única versión con wheels para Python 3.13).
- **No se usan marcadores ArUco.** El mapeo es directo de coordenadas normalizadas de MediaPipe al canvas del slide.
- **`holistic_landmarker.task`** se descarga en el primer arranque desde el bucket oficial de MediaPipe. No se versiona en el repo.
- Los modelos entrenados (`.h5`) y los datasets (`.npy`) **no se versionan** — cada usuario entrena los suyos.

---

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---|---|---|
| `AttributeError: module mediapipe has no attribute 'solutions'` | Versión de mediapipe moderna | El tracker ya usa Tasks API; asegúrate de estar en la rama actual |
| Backend no arranca, falta `.task` | Primera ejecución sin internet | Conecta y reinicia, o descarga manualmente a `core/models/holistic_landmarker.task` |
| LSTM predice siempre la misma clase | Dataset desbalanceado o <10 muestras/clase | Graba más muestras (~15–20/clase) y reentrena |
| Modelo se carga pero no predice | Versión `.h5` incompatible con `MODEL_ARCH_VERSION` | Borra `core/models/gestures_model_*.h5` y reentrena |

Descarga manual del modelo de MediaPipe (si la auto-descarga falla):

```bash
curl -L -o core/models/holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

---

## Estado del roadmap

- [x] Fase 1 — Cámara + MediaPipe + objeto siguiendo la palma
- [x] Fase 2 — Mapeo virtual (sin ArUco)
- [x] Fase 3 — LSTM dual (dominant + support) entrenado in-app
- [x] Fase 4 — Lectura básica de `.pptx`
- [x] Fase 5 — Electron + React + WebSocket
- [x] Fase 6 — Editor de objetos con drag & drop
- [x] Fase 7 — Física básica y render
- [ ] Fase 8 — Pulido y empaquetado `.exe` (en pausa)

---

## Licencia

MIT — Uso académico y personal.

---

GestDeck · Deep Learning 2026
