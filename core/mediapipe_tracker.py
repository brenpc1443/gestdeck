"""
mediapipe_tracker.py — Detección de landmarks con MediaPipe HolisticLandmarker.

Usa el nuevo API de MediaPipe Tasks (0.10.30+). El modelo `.task` se descarga
automáticamente la primera vez a `core/models/holistic_landmarker.task`.

HolisticLandmarker devuelve por frame:
    - 21 landmarks por mano izquierda (x, y, z normalizados 0-1)
    - 21 landmarks por mano derecha
    - 33 landmarks de pose corporal
    - (la cara y la segmentación se desactivan por rendimiento)

Cada landmark viene en coordenadas normalizadas del frame de la cámara:
    x ∈ [0, 1]  — horizontal, 0=izquierda del frame
    y ∈ [0, 1]  — vertical,   0=arriba del frame
    z ∈ ℝ       — profundidad relativa (negativo = hacia la cámara)

En Modo Virtual estas coordenadas se mapean DIRECTAMENTE al slide
digital mediante `virtual_mapper.VirtualMapper` — sin ArUco, sin homografía.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
MODEL_PATH = Path("core/models/holistic_landmarker.task")


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Descarga el modelo .task si no existe (≈50 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe_tracker] Descargando modelo holistic ({MODEL_URL}) …")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar el modelo MediaPipe. "
            f"Descárgalo manualmente a {path} desde:\n  {MODEL_URL}\n"
            f"Detalle: {e}"
        ) from e
    print(f"[mediapipe_tracker] Modelo guardado en {path}")
    return path


@dataclass
class HandLandmarks:
    """21 puntos (x, y, z) normalizados. shape = (21, 3)."""
    points: np.ndarray
    handedness: str              # "Left" o "Right"

    def flat_xy(self) -> np.ndarray:
        """Vector (42,) para el LSTM con dos componentes:

          indices 0-1  → posición ABSOLUTA de la muñeca en el frame.
                         Captura la TRAYECTORIA global de la mano (necesaria
                         para detectar swipes, empujes y cualquier gesto
                         donde la mano entera se desplaza).

          indices 2-41 → 20 landmarks (del índice base al meñique) expresados
                         RELATIVOS a la muñeca y escalados por la distancia
                         wrist → MCP_medio. Captura la FORMA de la mano de
                         manera invariante a posición y escala: un puño hecho
                         arriba o abajo, cerca o lejos, da la misma firma.

        Así el LSTM aprende por separado el movimiento global y la
        configuración de los dedos, y generaliza muchísimo mejor entre
        muestras grabadas en distintas zonas del frame.
        """
        p = self.points[:, :2].astype(np.float32)       # (21, 2)
        wrist = p[0].copy()
        mcp_medio = p[9]
        scale = float(np.linalg.norm(mcp_medio - wrist))
        if scale < 1e-6:
            scale = 1.0

        out = np.empty(42, dtype=np.float32)
        out[0] = wrist[0]
        out[1] = wrist[1]
        rel = (p[1:] - wrist) / scale                   # (20, 2)
        out[2:] = rel.flatten()
        return out

    @property
    def palm_center(self) -> np.ndarray:
        """Centro de la palma: promedio de landmarks 0, 5, 9, 13, 17.

        Siempre en coordenadas ABSOLUTAS del frame — lo usan el
        virtual_mapper (para colocar el cursor) y las reglas geométricas.
        """
        return self.points[[0, 5, 9, 13, 17]].mean(axis=0)


@dataclass
class TrackingResult:
    """Resultado de un frame procesado."""
    left_hand: Optional[HandLandmarks] = None
    right_hand: Optional[HandLandmarks] = None
    pose_landmarks: Optional[np.ndarray] = None   # (33, 4) x,y,z,visibility
    any_hand: bool = False
    detections_per_frame: int = 0

    def dominant(self, dominant: str) -> Optional[HandLandmarks]:
        if dominant.lower().startswith("r"):
            return self.right_hand
        return self.left_hand

    def support(self, dominant: str) -> Optional[HandLandmarks]:
        if dominant.lower().startswith("r"):
            return self.left_hand
        return self.right_hand


class Tracker:
    """Wrapper sobre HolisticLandmarker con control de frame-skip."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,       # mantenido por compatibilidad de API
        process_every_n: int = 2,        # 1 de cada N frames
    ):
        del model_complexity  # el nuevo API usa un solo asset
        model_path = ensure_model()
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            min_hand_landmarks_confidence=min_tracking_confidence,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_suppression_threshold=0.3,
            min_pose_landmarks_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)
        self._process_every_n = max(1, int(process_every_n))
        self._frame_counter = 0
        self._last_result: Optional[TrackingResult] = None
        self._epoch = time.time()
        self._last_ts_ms = -1

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    def process(self, bgr_frame: np.ndarray) -> TrackingResult:
        """Procesa un frame BGR (cv2) y devuelve el resultado de tracking.

        Implementa frame-skipping: entre frames procesados se reutiliza el
        último resultado (la interpolación del cliente suaviza la animación).
        """
        self._frame_counter += 1
        if self._frame_counter % self._process_every_n != 0 and self._last_result is not None:
            return self._last_result

        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video exige timestamps estrictamente crecientes en ms
        ts_ms = max(self._last_ts_ms + 1, int((time.time() - self._epoch) * 1000))
        self._last_ts_ms = ts_ms

        try:
            res = self._landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:
            print(f"[mediapipe_tracker] detect error: {e}")
            return self._last_result or TrackingResult()

        result = TrackingResult()

        lh = _first_hand(getattr(res, "left_hand_landmarks", None))
        rh = _first_hand(getattr(res, "right_hand_landmarks", None))
        pose = _first_hand(getattr(res, "pose_landmarks", None))

        if lh is not None:
            result.left_hand = HandLandmarks(points=_lm_to_array(lh, dim=3), handedness="Left")
        if rh is not None:
            result.right_hand = HandLandmarks(points=_lm_to_array(rh, dim=3), handedness="Right")
        if pose is not None:
            result.pose_landmarks = _lm_to_array(pose, dim=4)

        result.any_hand = result.left_hand is not None or result.right_hand is not None
        result.detections_per_frame = sum(
            x is not None for x in (result.left_hand, result.right_hand)
        )

        self._last_result = result
        return result


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _first_hand(value):
    """HolisticLandmarker puede devolver lista de NormalizedLandmark o lista anidada.
    Normalizamos a una única lista plana de landmarks (o None)."""
    if value is None:
        return None
    if not value:                       # lista vacía
        return None
    first = value[0]
    # Caso nested: value = [[lm, lm, ...]] (un grupo por persona/mano)
    if isinstance(first, (list, tuple)):
        return first if first else None
    # Caso plano: value = [lm, lm, ...]
    if hasattr(first, "x") and hasattr(first, "y"):
        return value
    return None


def _lm_to_array(lms, dim: int = 3) -> np.ndarray:
    """Convierte la lista de landmarks a np.ndarray (N, dim)."""
    if dim == 3:
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in lms],
        dtype=np.float32,
    )


# ----------------------------------------------------------------------
# Smoke test visual: python -m core.mediapipe_tracker
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    from core.camera import Camera

    with Camera() as cam, Tracker() as tracker:
        while True:
            f = cam.read()
            if f is None:
                continue
            res = tracker.process(f)
            for hand in (res.left_hand, res.right_hand):
                if hand is None:
                    continue
                h, w = f.shape[:2]
                for (x, y, _) in hand.points:
                    cv2.circle(f, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
            cv2.imshow("GestDeck tracker", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
