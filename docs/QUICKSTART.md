# Quickstart — GestDeck Modo Virtual

## Primera vez (instalación)

Necesitas:
- **Python 3.10+**
- **Node.js 18+**
- **Webcam**
- **LibreOffice** (opcional pero recomendado, para convertir .pptx con fidelidad). Descarga: https://es.libreoffice.org/

Desde la raíz del proyecto:

```powershell
# 1. Backend Python
python -m venv .venv
.\.venv\Scripts\activate          # (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Frontend Electron/React
cd app
npm install
cd ..
```

## Arrancar en modo desarrollo

**Opción A — script unificado:**

```powershell
# Windows
.\start.bat sessions\ejemplo_demo\config.json

# macOS / Linux
./start.sh sessions/ejemplo_demo/config.json
```

**Opción B — dos terminales:**

```powershell
# Terminal 1: backend
python -m core.main --config sessions\ejemplo_demo\config.json

# Terminal 2: frontend
cd app
npm run dev
```

Debería aparecer en la terminal del backend:
```
[ws] Servidor WebSocket en ws://127.0.0.1:8765
[main] Backend iniciado (cámara=ok, gestos=rule-based). Ctrl+C para salir.
```

Se abre la ventana **Editor**. Desde ahí:
1. *Cargar PPTX* o *Abrir sesión .json*.
2. Arrastra objetos desde la biblioteca al slide.
3. **▶ Iniciar presentación** → abre las ventanas Presentador (fullscreen) y Control.

## Atajos del Editor

| Tecla     | Acción                         |
|-----------|--------------------------------|
| `←` / `→` | Navegar entre slides           |
| `R`       | Reset de objetos a su origen   |
| `Supr`    | Eliminar objeto seleccionado   |
| `+` / `−` | Escalar objeto seleccionado    |
| Click     | Seleccionar objeto             |
| Click+drag| Mover objeto                   |
| Right-clk | Eliminar objeto                |

## Grabar tus propios gestos (Deep Learning completo)

El clasificador por reglas geométricas está activo por defecto. Para
mayor precisión entrena tu propio LSTM con tus manos:

```powershell
# Desde la raíz del proyecto, con .venv activada:
python -m training.record_gestures --gesto PUNO_CERRADO  --repeticiones 50
python -m training.record_gestures --gesto PALMA_ABIERTA --repeticiones 50
python -m training.record_gestures --gesto INDICE        --repeticiones 50
python -m training.record_gestures --gesto PELLIZCO      --repeticiones 50
python -m training.record_gestures --gesto PELLIZCO_INV  --repeticiones 50

python -m training.train_lstm
```

En la grabación:
- ESPACIO inicia una repetición (1 seg de countdown, 1 seg de captura).
- Repite el gesto ligeramente distinto cada vez (ángulos, distancias).
- `Q` aborta.

Al reiniciar `core.main`, la Control Window mostrará el badge `lstm`
en vez de `rule-based`, y los mensajes `modo_gesto` en el WS cambiarán
de `"rule-based"` a `"lstm"`.

## Problemas comunes

| Síntoma | Causa / fix |
|---|---|
| Presenter window sale negra | No has cargado una sesión en el Editor. Carga un PPTX o un config.json primero. |
| "Cámara no disponible" | Otra app la está usando. Ciérrala; el backend reintenta cada 3s. |
| PPTX no se convierte | Instala LibreOffice o `pip install python-pptx Pillow` para el fallback. |
| Gestos erráticos con reglas | Entrena tu LSTM — las reglas no distinguen bien PALMA_VERTICAL / EMPUJE. |
| Hand trackea pero objetos no se mueven | Verifica que el badge "Backend" esté verde en el Editor/Control. |
| Badge queda en `rule-based` | El backend no encontró `core/models/gestures_model.h5`. Entrena primero. |

## Compartir por videollamada (opcional)

En Zoom / Meet / Teams: *Share screen → selecciona la ventana "GestDeck — Presentador"*.
La audiencia verá solo slide + objetos respondiendo a tus gestos.
Mantén tu cámara del sistema encendida si quieres el PIP con tu cara (lo maneja la videollamada, no GestDeck).
