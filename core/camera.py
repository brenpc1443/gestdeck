"""
camera.py — Captura de video y preprocesamiento (CLAHE).

Responsabilidades:
    - Abrir la cámara web (OpenCV).
    - Capturar frames continuamente en un hilo independiente.
    - Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
      para robustecer la detección de MediaPipe ante iluminación pobre.
    - Exponer el último frame disponible a otros hilos de forma thread-safe.

En modo virtual no se necesita rectificar perspectiva (no hay proyector).
La cámara puede estar en cualquier ángulo siempre que vea las manos.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 0                # índice del dispositivo (0 = cámara por defecto)
    width: int = 1280
    height: int = 720
    fps_target: int = 30
    mirror: bool = True           # espejo horizontal: gesto derecha → objeto a la derecha del slide
    apply_clahe: bool = True
    clahe_clip: float = 2.5
    clahe_grid: int = 8


class Camera:
    """Captura thread-safe con preprocesamiento CLAHE opcional."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_raw: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_capture_ts = 0.0
        self._fps_ema = 0.0

        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip,
            tileGridSize=(self.config.clahe_grid, self.config.clahe_grid),
        )

    # ------------------------------------------------------------------
    # ciclo de vida
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.config.index, cv2.CAP_DSHOW if _is_windows() else 0)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara índice {self.config.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gestdeck-camera")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        """Devuelve el frame más reciente ya preprocesado (BGR uint8)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def read_raw(self) -> Optional[np.ndarray]:
        """Frame sin CLAHE ni mirror — útil para preview en la ventana de control."""
        with self._lock:
            if self._latest_raw is None:
                return None
            return self._latest_raw.copy()

    @property
    def fps(self) -> float:
        return self._fps_ema

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            raw = frame.copy()
            if self.config.mirror:
                frame = cv2.flip(frame, 1)

            if self.config.apply_clahe:
                frame = self._apply_clahe(frame)

            now = time.time()
            dt = now - self._last_capture_ts if self._last_capture_ts else 1 / 30
            self._last_capture_ts = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            # EMA suave
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst_fps if self._fps_ema else inst_fps

            with self._lock:
                self._latest_frame = frame
                self._latest_raw = raw

    def _apply_clahe(self, bgr: np.ndarray) -> np.ndarray:
        """CLAHE sobre el canal L de LAB — preserva color y mejora contraste local."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _is_windows() -> bool:
    import platform
    return platform.system().lower().startswith("win")


# ----------------------------------------------------------------------
# Pequeño smoke test: python -m core.camera
# ----------------------------------------------------------------------
if __name__ == "__main__":
    with Camera() as cam:
        print("Cámara abierta. Pulsa Q para salir.")
        while True:
            f = cam.read()
            if f is not None:
                cv2.putText(f, f"FPS: {cam.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("GestDeck camera", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
