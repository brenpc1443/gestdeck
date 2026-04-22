# Siguientes pasos — GestDeck

Guía práctica ordenada de lo que debes hacer desde ahora hasta una
presentación real con tus propios gestos y tu propia presentación.

---

## 0. Estado actual de tu instalación

Ya está hecho:

- [x] Entorno Python virtual en `.venv/` con todas las dependencias
      (tensorflow, mediapipe 0.10.33, opencv, websockets, python-pptx, Pillow).
- [x] `node_modules/` instalado en `app/` (Electron, React, Vite, Tailwind).
- [x] Modelo MediaPipe `holistic_landmarker.task` descargado en
      `core/models/`.
- [x] Backend verificado: arranca, abre WebSocket, acepta conexiones, emite
      `hello` + snapshot + `frame` a 30 Hz.
- [x] Sesión demo disponible en `sessions/ejemplo_demo/config.json`.

Falta (por ti):

- [ ] Probar la UI (Editor / Presenter / Control) en tu pantalla.
- [ ] Grabar tu dataset de gestos.
- [ ] Entrenar tu LSTM.
- [ ] Cargar tu `.pptx` real y configurar los objetos por slide.
- [ ] (Opcional) instalar LibreOffice si quieres render fiel de slides.

---

## 1. Primera prueba (sesión demo)

Abre una terminal en la raíz del proyecto y lanza el script unificado:

```powershell
.\start.bat sessions\ejemplo_demo\config.json
```

Se abrirán **dos terminales**:

1. *GestDeck Backend* — debería imprimir:
   ```
   [ws] Servidor WebSocket en ws://127.0.0.1:8765
   [main] Backend iniciado (cámara=ok, gestos=rule-based). Ctrl+C para salir.
   ```
2. *GestDeck Frontend* — `vite` + `electron`. Tras unos segundos abre la
   ventana **Editor**.

En el Editor:

1. Mira que en la barra lateral aparezcan las luces verdes (Backend,
   Cámara). Si la cámara está en uso por otra app, ciérrala — el backend
   reintenta cada 3 s.
2. Con la sesión demo cargada deberías ver 3 slides y 2 objetos en el
   slide 1 (círculo y flecha).
3. Muévelos con drag del ratón para comprobar que comunican con el backend.
4. Pulsa **▶ Iniciar presentación**. Se abren:
   - **Presenter** (fullscreen en monitor secundario si tienes, principal si no).
   - **Control** (ventana pequeña, *always-on-top*).

Pon la mano delante de la cámara y observa:
- El punto cian en Control sigue tu palma.
- El badge de gesto cambia según lo que hagas (con reglas geométricas es
  impreciso — es normal).
- Si cierras el puño cerca de un objeto, debería "agarrarlo".

Si todo esto funciona, la instalación está bien.

---

## 2. Grabar tu dataset de gestos

Ahora el objetivo es sustituir el fallback rule-based por una LSTM entrenada
**con tus manos**.

Necesitas grabar ≥50 repeticiones de cada gesto que quieras usar. Los gestos
estándar del proyecto son los 8 listados en [`PROYECTO.md`](./PROYECTO.md#6-gestos-y-acciones)
— puedes grabar los que te interesen (mínimo 2 distintos, recomendado ≥5).

Para cada gesto:

```powershell
# Activa el venv primero (si no está activo ya):
.\.venv\Scripts\activate

python -m training.record_gestures --gesto PUNO_CERRADO  --repeticiones 50
python -m training.record_gestures --gesto PALMA_ABIERTA --repeticiones 50
python -m training.record_gestures --gesto INDICE        --repeticiones 50
python -m training.record_gestures --gesto PELLIZCO      --repeticiones 50
python -m training.record_gestures --gesto PELLIZCO_INV  --repeticiones 50
# Opcionalmente:
python -m training.record_gestures --gesto PALMA_VERTICAL --repeticiones 50
python -m training.record_gestures --gesto EMPUJE         --repeticiones 50
```

Se abre una ventana con la cámara. Cada repetición dura **1 s de captura
tras 1 s de *countdown*** (total 2 s por rep).

**Tips para un buen dataset:**
- Graba con la **misma iluminación** que usarás en la exposición real
  (luz del salón, no luz tenue de la madrugada).
- Varía el **ángulo y la distancia** a la cámara en cada repetición
  (10 acercado, 10 alejado, 10 ligeramente rotado, etc.).
- Si vas a usar mano izquierda, añade `--mano left`.
- Evita fondos muy ruidosos; MediaPipe segmenta mejor con un fondo plano.
- Si te equivocas en una repetición, pulsa `Q` y vuelve a empezar
  (el `.npy` se sobrescribe).

Al terminar cada gesto, se guarda `training/dataset/<GESTO>.npy` y se
actualiza `training/dataset/gestures_index.json`.

---

## 3. Entrenar tu LSTM

Con al menos 2 gestos grabados (recomendado ≥5, ≥50 reps cada uno):

```powershell
python -m training.train_lstm
```

En CPU tarda típicamente **30-120 segundos**. Al final:

- `core/models/gestures_model.h5`       — tu modelo.
- `core/models/gestures_labels.json`    — lista de clases en orden.

Reinicia el backend (cierra y vuelve a lanzar `start.bat`). En la consola
verás ahora:

```
[gesture_classifier] LSTM cargada: 5 clases.
```

Y en la ventana Control, el banner amarillo de "rule-based" **desaparece**;
el badge de gesto pasa a mostrar `lstm` como fuente.

Si la precisión es baja: graba **más repeticiones** del gesto confundido
y re-entrena. No hace falta borrar el dataset — `record_gestures.py`
sobrescribe el `.npy` de cada gesto.

---

## 4. Preparar tu propia presentación

### 4a. (Recomendado) Instalar LibreOffice

El fallback `python-pptx + Pillow` genera slides muy estilizados (fondo
oscuro uniforme, título + bullets). Para que los PNGs sean **fieles** a
tu PPTX real, instala LibreOffice:

- Windows: https://es.libreoffice.org/descarga/libreoffice/
- GestDeck lo detecta automáticamente (busca `soffice` en el PATH y en
  `C:\Program Files\LibreOffice\program\`).

### 4b. Cargar tu PPTX en el Editor

1. Lanza `start.bat` (sin argumentos).
2. En el Editor pulsa **Cargar PPTX**.
3. Se convierte a PNGs en `sessions/<nombre>/slides/`.
4. El nombre de sesión se deriva del fichero (`MiExpo.pptx` → `miexpo`).

### 4c. Añadir objetos interactivos por slide

Por cada slide:

1. Navega al slide con las flechas `← →` o las miniaturas del pie.
2. Arrastra objetos desde la **biblioteca** (barra izquierda) al canvas:
   - Flecha, círculo, cuadrado (SVGs en `assets/objects/`).
   - Texto (te pide el contenido).
3. Click en un objeto = seleccionar → panel derecho muestra el inspector:
   posición, escala (slider), botón eliminar.
4. Click-drag mueve el objeto; `+` / `−` escalan el seleccionado;
   `Supr` lo elimina; `R` resetea todos los objetos a su origen.

Para añadir más tipos de objetos, suelta SVGs/PNGs en `assets/objects/` y
extiende el array `OBJECT_LIBRARY` en `app/src/windows/Editor.jsx`.

### 4d. Guardar la sesión

Pulsa **Guardar sesión**. Se escribe un `config.json` en
`sessions/<nombre>/config.json` con toda la configuración (objetos,
posiciones, calibración, mano dominante). Para retomarla después:

```powershell
.\start.bat sessions\<nombre>\config.json
```

---

## 5. Presentación real

1. Lanza `start.bat` con tu sesión.
2. En el Editor, verifica luces verdes y mano dominante.
3. (Opcional) **Calibrar área de gestos (3 s)** — te da 3 segundos para
   barrer el espacio en el que vas a gesticular; eso ajusta `active_region`
   en el mapper. Útil si estás lejos de la cámara y usas solo un cuadrante.
4. Pulsa **▶ Iniciar presentación**.
5. En Zoom / Meet / Teams:
   - *Share Screen → Window → GestDeck — Presentador*.
   - Mantén tu cámara del sistema encendida si quieres que se te vea por PIP
     (eso lo gestiona la videollamada, no GestDeck).
6. La ventana **Control** es solo para ti (no la compartas) — te muestra
   qué gesto detecta y el FPS.

Durante la exposición:

- Mano dominante delante de la cámara = controlas objetos.
- Mano de apoyo en bordes extremos (x > 0.85 o x < 0.15) = navega slides.
- Si se te congela un objeto sin que lo sueltes, haz `EMPUJE` o pulsa `R`
  en el Control para resetear.

---

## 6. Roadmap / Mejoras futuras

Orden sugerido si quieres seguir iterando:

- **Fase 8 — Empaquetado .exe.** `npm run build` en `app/` ejecuta
  `vite build && electron-builder`. La config está en `app/package.json`
  bajo `build`. Tendrías que validar permisos de webcam en el .exe firmado.
- **Más tipos de objetos.** Imágenes PNG del usuario, GIFs animados,
  fórmulas LaTeX renderizadas, gráficas en vivo.
- **Grabación de la exposición.** Capturar la ventana Presenter a MP4 con
  `ffmpeg` invocado desde main.js.
- **Threshold de confianza por gesto.** Actualmente hay un único
  `CONFIDENCE_THRESHOLD = 0.6`; gestos ambiguos se beneficiarían de
  umbrales distintos.
- **Cambio de objeto activo con `INDICE` sobre miniatura.** Útil para
  manipular objetos en capas.
- **WebGL / Three.js** si necesitas transparencias, iluminación o
  partículas (hoy DOM/CSS es suficiente).

---

## 7. Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| La ventana Presenter sale negra | No hay sesión cargada en el Editor. Carga un PPTX o un `config.json`. |
| Badge queda en `rule-based` | `core/models/gestures_model.h5` no existe. Graba + entrena (pasos 2 y 3). |
| "Cámara no disponible" en Control | Otra app la está usando. Ciérrala; el backend reintenta cada 3 s. |
| El PPTX no se convierte | Instala LibreOffice o `pip install python-pptx Pillow` (Pillow ya está instalado). |
| Gestos muy inestables con LSTM | Graba más reps del gesto confuso; varía ángulos/iluminación; re-entrena. |
| Frontend no conecta al backend | Verifica que `ws://127.0.0.1:8765` no esté ocupado por otro proceso. |
| Objetos se mueven con lag | Baja `process_every_n` a 1 en `core/main.py` (línea de `Tracker(process_every_n=2)`) si tu CPU aguanta. |
| Me confunde la mano espejo | En el Editor, alterna "mano dominante" entre derecha/izquierda; la cámara ya está en modo espejo por defecto. |
| Al cerrar Presenter, el backend sigue vivo | Es intencional. Cierra el Editor o mata las terminales `cmd` para parar todo. |

---

## 8. Comandos de referencia rápida

```powershell
# Activar venv
.\.venv\Scripts\activate

# Lanzar todo
.\start.bat
.\start.bat sessions\ejemplo_demo\config.json

# Backend aislado (para debugging)
python -m core.main
python -m core.main --config sessions\demo\config.json

# Frontend aislado (backend debe estar corriendo)
cd app
npm run dev

# Smoke test individual de cada módulo
python -m core.camera              # preview de webcam + CLAHE
python -m core.mediapipe_tracker   # overlay de landmarks
python -m core.websocket_server    # server en loop enviando ping
python -m core.slide_reader <pptx> [out_dir]

# Grabar un gesto
python -m training.record_gestures --gesto AGARRAR --repeticiones 50 --mano right

# Entrenar LSTM
python -m training.train_lstm

# Build de producción (Fase 8 — no testeado)
cd app
npm run build
```

---

GestDeck — Deep Learning 2026 — MIT
