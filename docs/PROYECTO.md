# GestDeck — Modo Virtual (Ghost Mode)

Documentación técnica del proyecto.

---

## 1. Qué es

GestDeck es una aplicación de escritorio que permite **manipular objetos
digitales sobre diapositivas mediante gestos naturales de las manos**, sin
proyector físico y sin marcadores ArUco.

La audiencia (por videollamada o monitor secundario) ve la diapositiva con
los objetos moviéndose en tiempo real, pero **no ve al expositor** — de ahí
el nombre "Ghost Mode" o "Modo Fantasma". Si el expositor quiere aparecer,
lo hace mediante el Picture-in-Picture de Zoom/Meet/Teams, no por GestDeck.

El proyecto combina:

- **Visión por computador** (MediaPipe Holistic — 21 landmarks por mano + pose).
- **Deep Learning** (LSTM entrenada con el dataset del propio usuario).
- **Física ligera** (inercia, magnetismo, gravedad en palma) sobre los objetos.
- **Electron + React** para el editor y la ventana que se comparte.

---

## 2. Arquitectura

```
┌─────────────────────────────────────┐           ┌─────────────────────────────────┐
│  Backend Python — core/             │           │  Frontend Electron — app/       │
│                                     │           │                                 │
│  camera.py  (OpenCV + CLAHE)        │           │  main.js      (procesos/IPC)    │
│        ↓                            │           │  preload.js   (bridge seguro)   │
│  mediapipe_tracker.py               │    WS     │                                 │
│     HolisticLandmarker (21+21+33)   │ ────────► │  Editor.jsx                     │
│        ↓                            │  127.0.0.1│  Presenter.jsx   (fullscreen)   │
│  virtual_mapper.py                  │   :8765   │  Control.jsx    (always-on-top) │
│     cámara → slide (identity/zoom)  │ ◄──────── │                                 │
│        ↓                            │    JSON   │  SlideCanvas    (CSS/DOM)       │
│  gesture_classifier.py              │           │  useWebSocket   (reconexión)    │
│     LSTM (+ rule-based fallback)    │           │  useObjects     (interpolación) │
│        ↓                            │           │  useSlideImages (estado IPC)    │
│  object_engine.py                   │           │                                 │
│     estados + física                │           │                                 │
│        ↓                            │           │                                 │
│  websocket_server.py                │           │                                 │
└─────────────────────────────────────┘           └─────────────────────────────────┘
```

### Flujo por frame (≈30 Hz)

1. `camera.py` captura un frame BGR, aplica CLAHE y lo deja en un buffer thread-safe.
2. `mediapipe_tracker.process(frame)` devuelve `left_hand`, `right_hand`, `pose_landmarks`
   (21/21/33 landmarks normalizados x,y,z).
3. `virtual_mapper.map_point(x, y)` mapea la palma de la mano dominante al
   espacio del slide `[0,1] × [0,1]`.
4. `gesture_classifier.predict(...)` clasifica el gesto con la LSTM (secuencia de
   30 frames = ~1 s) o con heurísticas si el modelo no está entrenado.
5. `object_engine.apply_gesture(gesto, conf, xy)` actualiza los objetos según el
   gesto (agarrar, soltar, escalar, flotar, devolver) aplicando física.
6. `websocket_server.send({tipo:"frame", …})` envía el estado completo del slide
   activo a todos los clientes Electron conectados.

### Regla de resiliencia del canal

Ningún extremo espera respuesta del otro. Si el WebSocket cae:
- Python sigue procesando y encolando; los clientes reconectan con
  *back-off* exponencial (≤5 s).
- Al reconectarse, cada cliente recibe `hello` + snapshot de estado
  (`modo_gesto`, `camara_ok`/`error_camara`).
- A partir del siguiente `frame` la sincronización continúa.

---

## 3. Componentes del backend (`core/`)

| Archivo | Responsabilidad |
|---|---|
| `camera.py`              | Captura thread-safe desde webcam, CLAHE opcional para robustez lumínica, EMA de FPS. |
| `mediapipe_tracker.py`   | Wrapper del nuevo `HolisticLandmarker` (Tasks API). Auto-descarga `holistic_landmarker.task` la primera vez. Frame-skipping para ahorrar CPU. |
| `virtual_mapper.py`      | 3 modos: `identity` (directo), `zoom` (recorte del área activa), `calibrated` (homografía de 4 esquinas). Sustituye al `aruco_mapper` del diseño original. |
| `gesture_classifier.py`  | LSTM de secuencia (30×42) con *fallback* heurístico basado en extensión de dedos + distancia pulgar/índice. Umbral de confianza configurable. |
| `object_engine.py`       | Ciclo de vida de `Objeto` con estados `EN_SLIDE / EN_MANO / FLOTANDO / CONGELADO`. Fricción, magnetismo suave, impulso de retorno al origen. |
| `slide_reader.py`        | Convierte `.pptx` a PNGs. Estrategia 1: LibreOffice → PDF → `pdftoppm`/PyMuPDF. Estrategia 2: fallback `python-pptx + Pillow` con render estilizado. |
| `websocket_server.py`    | Servidor asyncio en hilo aparte. API thread-safe: `send(dict)` + callback `on_message`. Callback `on_connect` reenvía snapshot a clientes tardíos. |
| `main.py`                | Orquestador. Monta los componentes, crea el hilo de percepción, maneja los 14 mensajes entrantes de Electron, reintenta cámara cada 3 s si falla. |

---

## 4. Componentes del frontend (`app/`)

### Procesos Electron

- **main.js** — crea hasta 3 ventanas (Editor / Presenter / Control), registra el
  protocolo `gestdeck://` (sirve ficheros del proyecto con path traversal
  bloqueado), expone IPC para diálogos nativos, lectura/escritura de
  `config.json`, invocación de `core.slide_reader` como subproceso y un
  **estado de slides compartido** entre ventanas.
- **preload.js** — bridge seguro (`contextBridge`) que expone `window.gestdeck`
  con los invokers al main process. Contiene también la URL del WebSocket.

### React

- **Editor.jsx** — Vista de configuración de sesión: carga PPTX, arrastra
  objetos de la biblioteca, los reposiciona con drag, los elimina con
  right-click. Atajos `← → R Supr + −`. Inspector lateral del objeto
  seleccionado. Tira de miniaturas por slide.
- **Presenter.jsx** — Ventana fullscreen (Ghost Mode: sin la cámara del
  expositor). Pinta el slide actual + los objetos que reaccionan a los gestos.
- **Control.jsx** — Ventana `alwaysOnTop` del expositor con preview, FPS,
  estado del gesto, indicador `lstm`/`rule-based` y avisos de errores
  de cámara.

### Hooks

- `useWebSocket.js`    — conexión y reconexión automáticas. Separa `frame`
  (alta frecuencia → `lastFrame`) de otros eventos (→ `lastEvent`).
- `useObjects.js`      — interpolación suave (lerp 0.35) de las posiciones
  de los objetos para 60 fps visuales aunque el backend envíe a ~30 fps.
- `useSlideImages.js`  — se suscribe al estado de slides compartido vía IPC
  (`gestdeck:slides-updated`). Editor → Presenter/Control.

### Rendering

- **SlideCanvas.jsx** — usa DOM + CSS transforms (no Three.js). Halos de
  color según el estado del objeto (cyan = en mano, morado = flotando,
  rojo tenue = congelado). Anillo de selección para el objeto activo en
  el Editor.
- **GestureBadge.jsx** — chip de color con el gesto actual, la confianza en
  porcentaje y la fuente (`lstm` / `rule-based`).

---

## 5. Protocolo WebSocket

Documento completo en [`websocket_protocol.md`](./websocket_protocol.md).

Resumen:

- Transport: `ws://127.0.0.1:8765`, JSON.
- **Python → Electron**: `hello`, `modo_gesto`, `camara_ok` | `error_camara`,
  `frame`, `cambio_slide`, `sesion_cargada`, `sesion_guardada`,
  `session_ready`, `reiniciado_ok`, `objeto_agregado`, `objeto_eliminado`,
  `calibracion_*`, `error`.
- **Electron → Python**: `reiniciar`, `cambiar_slide`, `iniciar_calibracion`,
  `finalizar`, `cargar_sesion`, `guardar_sesion`, `setup_session`,
  `agregar_objeto`, `mover_objeto`, `eliminar_objeto`, `escalar_objeto`,
  `set_mirror`, `set_mano_dominante`.

---

## 6. Gestos y acciones

| Gesto            | Acción sobre el objeto activo / sistema |
|------------------|----------------------------------------|
| `INDICE`         | Seleccionar el objeto bajo el dedo     |
| `PUNO_CERRADO`   | Agarrar y mover con la mano            |
| `PALMA_ABIERTA`  | El objeto reposa sobre la palma        |
| `PALMA_VERTICAL` | Flota frente al expositor              |
| `PELLIZCO`       | Zoom in (escala ×1.02 por frame)       |
| `PELLIZCO_INV`   | Zoom out (escala ×0.98)                |
| `EMPUJE`         | Devolver el objeto a su posición origen|
| `SIGUIENTE`      | Navegar al slide siguiente             |
| `ANTERIOR`       | Navegar al slide anterior              |
| `NINGUNO`        | Magnetismo suave si la mano está cerca |

La **mano dominante** controla los objetos. La **mano de apoyo** navega
entre slides si se mueve hacia los bordes extremos del frame (x > 0.85 o
x < 0.15) con cooldown de 1.2 s.

---

## 7. Decisiones de diseño clave

- **Sin ArUco.** El diseño original usaba 5 marcadores sobre el proyector
  con homografía. Modo Virtual mapea directamente coordenadas normalizadas
  de cámara → slide, eliminando problemas de oclusión e iluminación.
- **Backend / frontend desacoplados.** WebSocket en vez de IPC nativo permite
  que el backend corra aisladamente (útil para pruebas) y que el frontend
  sobreviva a un reinicio del backend (y viceversa).
- **Ghost Mode.** La ventana Presenter nunca muestra al expositor. Si se
  quiere PIP, lo pone Zoom/Meet.
- **LSTM entrenada por el usuario final.** Los gestos dependen mucho de
  anatomía y estilo. En vez de enviar un modelo pre-entrenado (genérico,
  pobre), se incluye un pipeline de grabación + entrenamiento reproducible
  en minutos.
- **Fallback heurístico.** Permite usar la app desde la primera ejecución,
  sin entrenar nada. El `fuente` en cada `frame` indica qué clasificador
  está activo.
- **Slides como fondo estático.** El .pptx se convierte una sola vez a PNGs
  por sesión (LibreOffice + pdftoppm/PyMuPDF). No se interpreta el .pptx
  en tiempo real — GestDeck dibuja solo los objetos interactivos encima.
- **DOM/CSS en vez de WebGL.** Para las ≤20 figuras típicas por slide, DOM
  es más simple de depurar, suficientemente fluido y accesible a herramientas
  del navegador. La arquitectura permite sustituirlo por Three.js sin tocar
  el resto.

---

## 8. Estructura de carpetas

```
GestDeck/
├── core/                          Backend Python
│   ├── __init__.py
│   ├── main.py                    Orquestador
│   ├── camera.py                  Webcam + CLAHE (thread-safe)
│   ├── mediapipe_tracker.py       HolisticLandmarker (Tasks API)
│   ├── virtual_mapper.py          Mapeo cámara → slide
│   ├── gesture_classifier.py      LSTM + rule-based
│   ├── object_engine.py           Estados + física de objetos
│   ├── slide_reader.py            .pptx → PNG
│   ├── websocket_server.py        Servidor WS asyncio
│   └── models/
│       ├── holistic_landmarker.task      (auto-descargado)
│       ├── gestures_model.h5             (generado tras entrenar)
│       └── gestures_labels.json          (generado tras entrenar)
│
├── training/                      Dataset + entrenamiento LSTM
│   ├── record_gestures.py         Graba secuencias 30×42 por gesto
│   ├── train_lstm.py              Entrena la red
│   └── dataset/                   .npy por gesto + gestures_index.json
│
├── app/                           Frontend Electron + React + Vite
│   ├── main.js                    Procesos + IPC
│   ├── preload.js                 Bridge seguro
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.jsx                Router por hash (editor|presenter|control)
│       ├── index.jsx
│       ├── windows/
│       │   ├── Editor.jsx
│       │   ├── Presenter.jsx
│       │   └── Control.jsx
│       ├── components/
│       │   ├── SlideCanvas.jsx
│       │   └── GestureBadge.jsx
│       ├── hooks/
│       │   ├── useWebSocket.js
│       │   ├── useObjects.js
│       │   └── useSlideImages.js
│       └── styles/globals.css
│
├── assets/                        Recursos estáticos
│   ├── objects/                   SVGs arrastrables (flecha, círculo, cuadrado)
│   └── samples/
│
├── sessions/                      Sesiones del usuario
│   └── ejemplo_demo/
│       ├── config.json
│       └── slides/                PNG por slide (tras cargar un PPTX)
│
├── docs/
│   ├── PROYECTO.md                Este documento
│   ├── SIGUIENTES_PASOS.md        Guía práctica desde ahora
│   ├── QUICKSTART.md              Arranque rápido
│   └── websocket_protocol.md      Protocolo completo
│
├── start.bat                      Launcher Windows
├── start.sh                       Launcher macOS/Linux
├── requirements.txt               Dependencias Python
└── README.md
```

---

## 9. Stack técnico

| Capa | Tecnología |
|---|---|
| Captura & CV           | OpenCV (CLAHE, flip, VideoCapture DSHOW en Windows) |
| Landmark detection     | MediaPipe 0.10.33 — HolisticLandmarker (Tasks API) |
| Deep Learning          | TensorFlow / Keras 3 (LSTM 128 → 64 → Dense 64 → softmax) |
| Backend                | Python 3.10+ (probado en 3.13), `websockets`, `asyncio`, `threading` |
| Renderizado .pptx      | LibreOffice headless + pdftoppm / PyMuPDF · fallback python-pptx + Pillow |
| Frontend               | Electron 28, React 18, Vite 5, TailwindCSS 3 |
| Comunicación           | WebSocket JSON ~30 Hz (frames) + eventos puntuales |

---

## 10. Estado de las 8 fases

- [x] **Fase 1** — Cámara + MediaPipe + objeto siguiendo la palma.
- [x] **Fase 2** — Mapeo virtual (sin ArUco).
- [x] **Fase 3** — Scaffolding LSTM + fallback rule-based.
- [x] **Fase 4** — Lectura de .pptx (LibreOffice + fallback PIL).
- [x] **Fase 5** — Electron + React + WebSocket.
- [x] **Fase 6** — Editor con drag&drop, atajos, inspector, miniaturas.
- [x] **Fase 7** — Física de objetos + halos por estado (CSS/DOM).
- [ ] **Fase 8** — Empaquetado `.exe` con electron-builder (config preparada,
  build final no testeado).

---

GestDeck · Deep Learning 2026 · MIT
