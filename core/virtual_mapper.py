"""
virtual_mapper.py — Mapeo directo cámara → slide en Modo Virtual.

Reemplaza al aruco_mapper.py del diseño original. Como en modo virtual
no hay proyector físico, las coordenadas normalizadas de MediaPipe
(ya en [0, 1]) se mapean DIRECTAMENTE al espacio del slide con un
ajuste simple:

    1. Opción 'identidad'  — x_slide = x_cam, y_slide = y_cam.
       Útil cuando la cámara enmarca al presentador frontalmente.

    2. Opción 'zoom'       — recorta una sub-región de la cámara como
       el "área activa" y la re-normaliza a [0, 1]. Esto permite usar
       solo la zona donde el presentador gesticula (p.ej. el cuadrante
       central) y duplica la sensibilidad.

    3. Opción 'calibrada'  — el presentador define 4 puntos de la
       cámara (esquinas del área activa) en una calibración rápida
       de 3 segundos. Se aplica una transformación afín/perspectiva.

En todos los casos la salida es (x, y) ∈ [0, 1] × [0, 1] con origen
en la esquina superior izquierda del slide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class MapperConfig:
    mode: str = "identity"                  # "identity" | "zoom" | "calibrated"
    active_region: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # (x0, y0, x1, y1) en coordenadas normalizadas de la cámara.
    # Solo se usa en modo 'zoom' y 'calibrated'.
    corners_camera: Optional[list] = None   # [[x,y], x4] — usado en 'calibrated'
    sensitivity: float = 1.0                # multiplicador de rango
    invert_y: bool = False                  # si tu cámara está al revés
    min_clamp: float = -0.1                 # permite un poco fuera del slide para suavidad
    max_clamp: float = 1.1


class VirtualMapper:
    """Convierte coordenadas normalizadas de cámara a coordenadas de slide."""

    def __init__(self, config: Optional[MapperConfig] = None):
        self.config = config or MapperConfig()
        self._homography: Optional[np.ndarray] = None
        if self.config.mode == "calibrated" and self.config.corners_camera is not None:
            self._build_homography()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        """Mapea un punto de la cámara (x, y) ∈ [0,1] al espacio del slide."""
        if self.config.mode == "identity":
            sx, sy = x, y
        elif self.config.mode == "zoom":
            sx, sy = self._map_zoom(x, y)
        elif self.config.mode == "calibrated":
            sx, sy = self._map_homography(x, y)
        else:
            raise ValueError(f"Modo desconocido: {self.config.mode}")

        # Aplicar sensibilidad (centrada en 0.5)
        if self.config.sensitivity != 1.0:
            sx = 0.5 + (sx - 0.5) * self.config.sensitivity
            sy = 0.5 + (sy - 0.5) * self.config.sensitivity

        if self.config.invert_y:
            sy = 1.0 - sy

        # Clamp suave
        sx = max(self.config.min_clamp, min(self.config.max_clamp, sx))
        sy = max(self.config.min_clamp, min(self.config.max_clamp, sy))
        return float(sx), float(sy)

    def map_hand(self, hand_points: np.ndarray) -> np.ndarray:
        """Mapea un array de 21 landmarks (21,3) → (21,2) en espacio slide."""
        out = np.zeros((len(hand_points), 2), dtype=np.float32)
        for i, (x, y, _z) in enumerate(hand_points):
            out[i] = self.map_point(float(x), float(y))
        return out

    # ------------------------------------------------------------------
    # Calibración en caliente (estilo "párate en el centro" simplificado)
    # ------------------------------------------------------------------
    def calibrate_from_samples(self, samples: np.ndarray) -> None:
        """Dada una nube de puntos (N, 2) recogida mientras el usuario
        'barre' su área de gestos, ajusta active_region para que encaje.

        samples: coordenadas de la cámara normalizadas [0, 1].
        """
        if len(samples) < 10:
            return
        x0 = float(np.percentile(samples[:, 0], 5))
        x1 = float(np.percentile(samples[:, 0], 95))
        y0 = float(np.percentile(samples[:, 1], 5))
        y1 = float(np.percentile(samples[:, 1], 95))
        # Expandir un 10% para dar margen
        dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        self.config.active_region = (
            max(0.0, x0 - dx),
            max(0.0, y0 - dy),
            min(1.0, x1 + dx),
            min(1.0, y1 + dy),
        )
        self.config.mode = "zoom"

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "mode": self.config.mode,
            "active_region": list(self.config.active_region),
            "corners_camera": self.config.corners_camera,
            "sensitivity": self.config.sensitivity,
            "invert_y": self.config.invert_y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VirtualMapper":
        cfg = MapperConfig(
            mode=d.get("mode", "identity"),
            active_region=tuple(d.get("active_region", (0.0, 0.0, 1.0, 1.0))),
            corners_camera=d.get("corners_camera"),
            sensitivity=float(d.get("sensitivity", 1.0)),
            invert_y=bool(d.get("invert_y", False)),
        )
        return cls(cfg)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VirtualMapper":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _map_zoom(self, x: float, y: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.config.active_region
        w = max(1e-6, x1 - x0)
        h = max(1e-6, y1 - y0)
        return (x - x0) / w, (y - y0) / h

    def _map_homography(self, x: float, y: float) -> Tuple[float, float]:
        if self._homography is None:
            return x, y
        pt = np.array([[x, y, 1.0]]).T
        out = self._homography @ pt
        out /= out[2, 0]
        return float(out[0, 0]), float(out[1, 0])

    def _build_homography(self) -> None:
        import cv2
        src = np.array(self.config.corners_camera, dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self._homography, _ = cv2.findHomography(src, dst)


# ----------------------------------------------------------------------
# Demo: test básico
# ----------------------------------------------------------------------
if __name__ == "__main__":
    m = VirtualMapper()
    print("identity:", m.map_point(0.3, 0.7))
    m2 = VirtualMapper(MapperConfig(mode="zoom", active_region=(0.2, 0.2, 0.8, 0.8)))
    print("zoom:    ", m2.map_point(0.5, 0.5))   # centro → centro
    print("zoom:    ", m2.map_point(0.2, 0.2))   # esquina → (0, 0)
    print("zoom:    ", m2.map_point(0.8, 0.8))   # esquina → (1, 1)
