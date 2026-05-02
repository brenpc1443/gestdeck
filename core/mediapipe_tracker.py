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
        """Compat: asume frame mirroreado (mirror=True). Usar `for_user`
        cuando se conozca el mirror real del frame."""
        if dominant.lower().startswith("r"):
            return self.right_hand
        return self.left_hand

    def support(self, dominant: str) -> Optional[HandLandmarks]:
        """Compat: ver `dominant`. Usar `for_user` para resultados correctos."""
        if dominant.lower().startswith("r"):
            return self.left_hand
        return self.right_hand

    def for_user(
        self,
        user_dominance: str,
        mirror: bool,
    ) -> tuple[Optional[HandLandmarks], Optional[HandLandmarks]]:
        """Devuelve (mano_dominante, mano_apoyo) según la posición real
        en el frame y la configuración de mirror.

        El tracker slotea por x (left_hand=baja x, right_hand=alta x).
        Bajo mirror=True el usuario se ve "como en un espejo" → su mano
        derecha física está a alta x → right_hand slot. Bajo mirror=False
        el frame está al revés → la derecha física está a baja x →
        left_hand slot.

        Tabla:
            mirror=True,  derecha    → dom=right_hand, sup=left_hand
            mirror=True,  izquierda  → dom=left_hand,  sup=right_hand
            mirror=False, derecha    → dom=left_hand,  sup=right_hand
            mirror=False, izquierda  → dom=right_hand, sup=left_hand
        """
        is_right = user_dominance.lower().startswith("r") or user_dominance.lower().startswith("d")
        # XNOR: dom va al slot de "alta x" cuando is_right==mirror
        dom_high_x = (is_right and mirror) or (not is_right and not mirror)
        if dom_high_x:
            return self.right_hand, self.left_hand
        return self.left_hand, self.right_hand


class Tracker:
    """Wrapper sobre HolisticLandmarker con control de frame-skip."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.65,
        model_complexity: int = 1,       # mantenido por compatibilidad de API
        process_every_n: int = 2,        # 1 de cada N frames
        mirror: bool = True,             # frame espejado (default en GestDeck)
    ):
        del model_complexity  # el nuevo API usa un solo asset
        # Flag mutable: main.py lo sincroniza con camera.config.mirror cuando
        # cambia. La asignación de slots por handedness depende de él.
        self.mirror = mirror
        model_path = ensure_model()
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            # Subido a 0.65 para que MP no devuelva landmarks de baja
            # confianza cuando hay ambigüedad (dedos hacia cámara, manos
            # ocluyendo, motion blur). Preferimos "no hay mano" a
            # "landmarks en lugares imposibles".
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
        # Estabilización de identidad: posición del palm_center de cada
        # slot en el frame anterior. Evita que MediaPipe intercambie las
        # etiquetas L/R cuando las manos están cerca o los dedos apuntan
        # a la cámara (la identidad la decidimos por continuidad espacial,
        # no por la etiqueta de MP que es inestable).
        self._last_left_palm: Optional[np.ndarray] = None
        self._last_right_palm: Optional[np.ndarray] = None
        self._gap_left: int = 0
        self._gap_right: int = 0
        # Tras este número de frames sin ver una mano, descartamos su última
        # posición. ~0.4s a 30 FPS efectivos del tracker.
        self._max_gap_frames: int = 12
        # Bridging: cuando MP pierde una mano hasta `_bridge_max` frames
        # consecutivos, devolvemos los últimos landmarks válidos en vez
        # de None. Enmascara los micro-cortes (dedos al frente, pellizco
        # apretado, motion blur) sin ocultar pérdidas reales (>130ms).
        self._last_left_hand: Optional[HandLandmarks] = None
        self._last_right_hand: Optional[HandLandmarks] = None
        self._bridge_max: int = 4
        # Solo bridgear un slot si tuvo al menos `_bridge_min_streak` frames
        # reales consecutivos antes de perderse. Sin esto, una detección
        # phantom de un solo frame (MP confundiendo un brazo con mano)
        # generaría un fantasma que persiste 130 ms.
        self._bridge_min_streak: int = 2
        self._eligible_bridge_left: bool = False
        self._eligible_bridge_right: bool = False
        self._streak_left: int = 0
        self._streak_right: int = 0
        # Span máximo permitido para wrist→MCP medio (en coords normalizadas
        # del frame). Una mano a distancia normal ocupa 5-10% del frame; al
        # acercarla deliberadamente a la cámara puede ocupar hasta ~45%. Por
        # encima de 0.5 ya casi siempre es un brazo o el cuerpo.
        self._max_hand_span: float = 0.5

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
            cand = HandLandmarks(points=_lm_to_array(lh, dim=3), handedness="Left")
            if self._is_plausible_hand(cand):
                result.left_hand = cand
        if rh is not None:
            cand = HandLandmarks(points=_lm_to_array(rh, dim=3), handedness="Right")
            if self._is_plausible_hand(cand):
                result.right_hand = cand
        if pose is not None:
            result.pose_landmarks = _lm_to_array(pose, dim=4)

        # --- Estabilización de identidad ---------------------------------
        # Asignamos cada detección al slot del tracker usando la handedness
        # ANATÓMICA que MP reporta + el flag mirror del frame. Esto evita
        # que una mano nueva entrante sea adoptada por el slot de la mano
        # anterior cuando ambas están en posiciones similares (bug típico
        # al cambiar de dominante a apoyo rápidamente).
        left_real, right_real = self._stabilize_identity(
            result.left_hand, result.right_hand,
        )

        # --- Bridging de huecos cortos -----------------------------------
        # Si un slot quedó vacío pero ese mismo slot tenía mano hace pocos
        # frames Y tenía una racha real consistente (≥_bridge_min_streak),
        # reusamos los últimos landmarks válidos. La condición de racha
        # filtra phantoms de un solo frame que generaban manos fantasma
        # estiradas. Lo importante es que `_update_palm_trackers` reciba
        # el estado REAL (no el bridged), para que el contador de gap
        # avance y eventualmente el bridge expire cuando la mano se haya
        # ido de verdad.
        left_out = left_real
        right_out = right_real
        if (
            left_real is None
            and self._last_left_hand is not None
            and self._gap_left < self._bridge_max
            and self._eligible_bridge_left
        ):
            left_out = self._last_left_hand
        if (
            right_real is None
            and self._last_right_hand is not None
            and self._gap_right < self._bridge_max
            and self._eligible_bridge_right
        ):
            right_out = self._last_right_hand

        result.left_hand = left_out
        result.right_hand = right_out
        result.any_hand = left_out is not None or right_out is not None
        result.detections_per_frame = sum(
            x is not None for x in (left_out, right_out)
        )
        self._update_palm_trackers(left_real, right_real)
        # Cache de landmarks para bridging — sólo cuando MP los devolvió
        # de verdad (no propagar bridges sobre bridges).
        if left_real is not None:
            self._last_left_hand = left_real
        if right_real is not None:
            self._last_right_hand = right_real

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    def _stabilize_identity(
        self,
        lh: Optional[HandLandmarks],
        rh: Optional[HandLandmarks],
    ) -> tuple[Optional[HandLandmarks], Optional[HandLandmarks]]:
        """Asigna detecciones a los slots del tracker (left_hand=baja x,
        right_hand=alta x) usando la handedness ANATÓMICA que reporta
        MediaPipe (`left_hand_landmarks` vs `right_hand_landmarks`),
        corregida por el flag `mirror` del frame.

        Cuando el frame está espejado (mirror=True), la chirality de MP se
        invierte respecto al usuario: la mano que MP llama "Left" es en
        realidad la mano DERECHA física del usuario (que aparece a alta x
        en el frame mostrado), y viceversa. Sin mirror, anatómica y posición
        coinciden directamente.

        Tabla de mapeo (handedness MP → slot tracker):
            mirror=True : MP "Left" → slot right_hand (alta x);
                          MP "Right" → slot left_hand (baja x).
            mirror=False: MP "Left" → slot left_hand (baja x);
                          MP "Right" → slot right_hand (alta x).

        Identificar las manos por anatomía (no por proximidad al último
        palm visto) evita que una mano nueva entrante sea adoptada por el
        slot vacío de la mano que se acaba de retirar — bug que aparecía al
        cambiar de dominante a apoyo rápidamente.

        Defensa contra glitches de chirality (manos juntas, dedos a cámara):
        si MP da ambas manos pero sus posiciones x contradicen claramente
        la convención esperada por mirror, asumimos que MP confundió las
        etiquetas y las swappeamos.
        """
        if lh is None and rh is None:
            return None, None

        # Detección de chirality swappeada por MP (caso raro): si tenemos
        # ambas manos, sus posiciones x deberían estar ordenadas según el
        # mirror. Si no, MP se confundió → corregir.
        if lh is not None and rh is not None:
            lh_x = float(lh.palm_center[0])
            rh_x = float(rh.palm_center[0])
            if self.mirror and lh_x < rh_x:
                # Espejado: MP "Left" debería estar a mayor x. Swap.
                lh, rh = rh, lh
            elif not self.mirror and lh_x > rh_x:
                # Sin espejo: MP "Left" debería estar a menor x. Swap.
                lh, rh = rh, lh

        # Mapeo handedness → slot del tracker.
        if self.mirror:
            return rh, lh
        return lh, rh

    def _update_palm_trackers(
        self,
        left: Optional[HandLandmarks],
        right: Optional[HandLandmarks],
    ) -> None:
        """Actualiza la posición conocida de cada slot, su racha de
        detecciones reales consecutivas y la elegibilidad para bridging.

        - Tras max_gap_frames sin ver una mano, su última posición se
          olvida (slot libre).
        - La elegibilidad para bridge se enciende cuando la racha llega
          a _bridge_min_streak. Permanece encendida durante el gap y se
          reevalúa cuando el slot vuelve a tener detección.
        """
        # left
        if left is not None:
            if self._gap_left > 0:
                # Transición de gap a real: la racha empieza de cero
                self._streak_left = 0
            self._streak_left += 1
            self._gap_left = 0
            self._eligible_bridge_left = self._streak_left >= self._bridge_min_streak
            self._last_left_palm = left.palm_center.copy()
        else:
            self._gap_left += 1
            if self._gap_left > self._max_gap_frames:
                self._last_left_palm = None
                self._eligible_bridge_left = False
        # right
        if right is not None:
            if self._gap_right > 0:
                self._streak_right = 0
            self._streak_right += 1
            self._gap_right = 0
            self._eligible_bridge_right = self._streak_right >= self._bridge_min_streak
            self._last_right_palm = right.palm_center.copy()
        else:
            self._gap_right += 1
            if self._gap_right > self._max_gap_frames:
                self._last_right_palm = None
                self._eligible_bridge_right = False

    def _is_plausible_hand(self, hand: HandLandmarks) -> bool:
        """Filtro de sanidad: rechaza landmarks claramente erróneos.

        Cuando MediaPipe confunde un brazo, el cuerpo o un objeto con
        forma de mano, devuelve landmarks "estirados": la distancia
        wrist→MCP medio supera con creces el tamaño normal de una mano.
        Una mano humana a distancia razonable de la cámara ocupa 5-10%
        del frame; >25% es chatarra y debe descartarse antes de slottear.
        """
        p = hand.points[:, :2]
        span = float(np.linalg.norm(p[9] - p[0]))  # wrist (0) → MCP medio (9)
        return 0.01 < span < self._max_hand_span


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
