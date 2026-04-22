# Protocolo WebSocket — GestDeck Modo Virtual

Canal bidireccional Python ⇄ Electron sobre `ws://127.0.0.1:8765`.
Todos los mensajes son JSON.

## Regla general

Ningún extremo espera respuesta del otro. Si el canal cae:
- Python sigue procesando y encolando el último estado.
- Electron congela los objetos en su última posición conocida.
- Al reconectar, el cliente recibe primero un mensaje `hello` y después
  un *snapshot* de eventos de estado (`modo_gesto`, `camara_ok` /
  `error_camara`). A partir del siguiente frame la sincronización
  continúa normal.

---

## Python → Electron

### `hello` (al conectarse el cliente)

```json
{ "tipo": "hello", "version": "0.1.0", "mode": "ghost" }
```

### Snapshot inmediato tras `hello`

El backend envía también el estado actual para que clientes que se
conectan tarde vean la situación real:

```json
{ "tipo": "modo_gesto", "fuente": "lstm" }
{ "tipo": "camara_ok" }
```

o bien, si la cámara no está disponible:

```json
{ "tipo": "error_camara", "detalle": "No se pudo abrir la cámara índice 0" }
```

### `frame` (≈30 Hz)

Estado completo del slide actual.

```json
{
  "tipo": "frame",
  "gesto": "PUNO_CERRADO",
  "confianza": 0.92,
  "fuente": "lstm",
  "mano_detectada": true,
  "mano": { "x": 0.47, "y": 0.58 },
  "objeto_activo": "cpu_icon",
  "estado_objeto": "EN_MANO",
  "slide_actual": 1,
  "total_slides": 3,
  "iluminacion_ok": true,
  "camera_fps": 28.4,
  "objetos": [
    {
      "id": "cpu_icon",
      "tipo": "svg",
      "archivo": "assets/objects/circle.svg",
      "posicion_original": [0.3, 0.5],
      "posicion_actual": [0.47, 0.58],
      "escala": 1.0,
      "rotacion": 0.0,
      "slide_origen": 1,
      "slide_actual": 1,
      "estado": "EN_MANO",
      "z_index": 1,
      "velocidad": [0.0, 0.0],
      "meta": {}
    }
  ]
}
```

**Nota** — campos `aruco_ok` y `aruco_visibles` del protocolo original
se eliminan en Modo Virtual.

### Eventos especiales

```json
{ "tipo": "calibracion_ok",       "active_region": [0.1, 0.15, 0.9, 0.85] }
{ "tipo": "calibracion_fallida" }
{ "tipo": "calibracion_en_curso" }
{ "tipo": "luz_baja" }
{ "tipo": "cambio_slide",         "slide": 2 }
{ "tipo": "objeto_soltado",       "objeto": "cpu_icon", "slide": 2, "x": 0.4, "y": 0.5 }
{ "tipo": "sesion_cargada",       "slides": 3, "nombre": "demo", "dominante": "derecha" }
{ "tipo": "sesion_guardada",      "path": "sessions/demo/config.json" }
{ "tipo": "session_ready",        "slides": 3 }
{ "tipo": "reiniciado_ok" }
{ "tipo": "objeto_agregado",      "objeto": { /* Objeto */ } }
{ "tipo": "objeto_eliminado",     "id": "obj_abc123" }
{ "tipo": "mano_dominante_set",   "value": "derecha" }
{ "tipo": "modo_gesto",           "fuente": "lstm" | "rule-based" }
{ "tipo": "camara_ok" }
{ "tipo": "error_camara",         "detalle": "..." }
{ "tipo": "error",                "detalle": "...", "mensaje_original": "mover_objeto" }
```

---

## Electron → Python

```json
{ "tipo": "reiniciar" }
{ "tipo": "cambiar_slide",       "slide": 2 }
{ "tipo": "iniciar_calibracion" }
{ "tipo": "finalizar" }
{ "tipo": "cargar_sesion",       "path": "sessions/ejemplo_demo/config.json" }
{ "tipo": "guardar_sesion",      "path": "sessions/demo/config.json" }
{ "tipo": "setup_session",       "nombre": "demo", "total_slides": 3,
                                 "pptx_path": "...", "slides_dir": "sessions/demo/slides" }
{ "tipo": "agregar_objeto",      "slide": 1, "objeto": { /* Objeto */ } }
{ "tipo": "mover_objeto",        "id": "obj_1", "x": 0.4, "y": 0.5, "update_origin": true }
{ "tipo": "eliminar_objeto",     "id": "obj_1" }
{ "tipo": "escalar_objeto",      "id": "obj_1", "escala": 1.2 }
{ "tipo": "set_mirror",          "value": true }
{ "tipo": "set_mano_dominante",  "value": "derecha" | "izquierda" }
```

## Estados de objeto

| Estado     | Significado                               |
|------------|-------------------------------------------|
| `EN_SLIDE` | En su posición normal en el slide         |
| `EN_MANO`  | Siendo manipulado por la mano             |
| `FLOTANDO` | Suspendido delante del expositor          |
| `CONGELADO`| Mano perdida sin gesto de soltar          |

## Gestos base (clase → acción)

| Gesto                   | Acción                         |
|-------------------------|--------------------------------|
| `INDICE`                | Seleccionar objeto             |
| `PUNO_CERRADO`          | Agarrar y mover                |
| `PALMA_ABIERTA`         | Reposar en la palma            |
| `PALMA_VERTICAL`        | Objeto flota                   |
| `PELLIZCO`              | Zoom in                        |
| `PELLIZCO_INV`          | Zoom out                       |
| `EMPUJE`                | Devolver a origen              |
| `SIGUIENTE` / `ANTERIOR`| Cambiar slide                  |
| `NINGUNO`               | Sin acción (magnetismo suave)  |
