"""
virtual_mapper.py — Mapeo cámara → slide en Modo Virtual.

En modo virtual no hay proyector físico: las coordenadas normalizadas que
entrega MediaPipe (ya en [0, 1]) se mapean al espacio del slide ajustando
una "región activa" del frame que se estira a las dimensiones del slide.

Hay dos modos:

    1. 'identity'  — passthrough x_slide = x_cam, y_slide = y_cam.
       Útil cuando la cámara enmarca al presentador frontalmente y el
       usuario quiere control 1:1.

    2. 'adaptive'  — la región activa se ajusta sola al span aparente
       de la mano (proxy de distancia a la cámara). Mano lejos → región
       estrecha y centrada para que el alcance lateral cubra todo el
       slide; mano cerca → región amplia. Es el modo por defecto: el
       usuario no necesita calibrar nada.

En ambos casos la salida es (x, y) ∈ [0, 1] × [0, 1] con origen en la
esquina superior izquierda del slide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class MapperConfig:
    mode: str = "adaptive"                  # "identity" | "adaptive"
    active_region: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # (x0, y0, x1, y1) en coordenadas normalizadas de la cámara. En modo
    # 'adaptive' se actualiza dinámicamente según el span aparente de la mano.
    sensitivity: float = 1.0                # multiplicador de rango
    invert_y: bool = False                  # si tu cámara está al revés
    min_clamp: float = -0.1                 # permite un poco fuera del slide para suavidad
    max_clamp: float = 1.1
    # --- Modo adaptive ----------------------------------------------------
    # Span aparente de la mano (||wrist → MCP_medio|| en coords del frame)
    # de referencia para los extremos de distancia. La región activa se
    # interpola entre estos dos puntos según el span observado en runtime.
    adaptive_span_far: float = 0.04         # mano lejos → ocupa ~4% del frame
    adaptive_span_near: float = 0.15        # mano cerca → ocupa ~15%+ del frame
    adaptive_margin_far: float = 0.28       # margen lateral cuando lejos (región [0.28, 0.72])
    adaptive_margin_near: float = 0.08      # margen lateral cuando cerca (región [0.08, 0.92])
    adaptive_ema_alpha: float = 0.08        # suavizado del span observado (0=congelado, 1=sin filtro)


class VirtualMapper:
    """Convierte coordenadas normalizadas de cámara a coordenadas de slide."""

    def __init__(self, config: Optional[MapperConfig] = None):
        self.config = config or MapperConfig()
        # Estado del modo adaptive: span suavizado por EMA. None = sin
        # historia todavía; el primer frame válido lo inicializa directo
        # para evitar un transitorio largo desde 0.
        self._adaptive_span_ema: Optional[float] = None
        if self.config.mode == "adaptive":
            # Arrancamos con la región media (margen entre near y far) para
            # que el primer frame tenga un mapeo razonable antes de recibir
            # `update_from_hand_span`.
            self._set_active_region_from_margin(
                (self.config.adaptive_margin_far + self.config.adaptive_margin_near) / 2.0
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        """Mapea un punto de la cámara (x, y) ∈ [0,1] al espacio del slide."""
        if self.config.mode == "identity":
            sx, sy = x, y
        elif self.config.mode == "adaptive":
            sx, sy = self._map_zoom(x, y)
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
    # Modo adaptive: ajusta la región activa según la distancia de la mano
    # ------------------------------------------------------------------
    def update_from_hand_span(self, span: float) -> None:
        """Ajusta `active_region` según el span aparente de la mano.

        `span` es la distancia normalizada wrist→MCP_medio en coordenadas
        del frame. Cuando la mano está cerca de la cámara ocupa más espacio
        (span grande) → el usuario tiene rango amplio en el frame y podemos
        usar una región activa amplia. Cuando está lejos (span pequeño) →
        rango efectivo limitado, región más estrecha y centrada para que
        el alcance lateral siga cubriendo todo el slide sin que el usuario
        tenga que estirarse.

        Suavizamos con un EMA para evitar saltos cuando la mano cambia de
        gesto (PUNO_CERRADO encoge el span; PALMA lo expande). Si el modo
        no es 'adaptive' esta llamada no hace nada.
        """
        if self.config.mode != "adaptive":
            return
        if span is None or span <= 0:
            return

        # EMA del span observado.
        a = float(self.config.adaptive_ema_alpha)
        if self._adaptive_span_ema is None:
            self._adaptive_span_ema = float(span)
        else:
            self._adaptive_span_ema = (1 - a) * self._adaptive_span_ema + a * float(span)

        # Lerp del margen entre los extremos near/far. t=0 → far, t=1 → near.
        s_far = float(self.config.adaptive_span_far)
        s_near = float(self.config.adaptive_span_near)
        denom = max(1e-6, s_near - s_far)
        t = (self._adaptive_span_ema - s_far) / denom
        t = max(0.0, min(1.0, t))
        margin = (
            self.config.adaptive_margin_far * (1 - t)
            + self.config.adaptive_margin_near * t
        )
        self._set_active_region_from_margin(float(margin))

    def _set_active_region_from_margin(self, margin: float) -> None:
        """Construye una región activa centrada con el margen lateral dado."""
        margin = max(0.0, min(0.49, margin))
        self.config.active_region = (margin, margin, 1.0 - margin, 1.0 - margin)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "mode": self.config.mode,
            "active_region": list(self.config.active_region),
            "sensitivity": self.config.sensitivity,
            "invert_y": self.config.invert_y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VirtualMapper":
        # Sesiones viejas pueden traer "zoom" o "calibrated" — ya no
        # existen, todo se trata como adaptive.
        mode = d.get("mode", "adaptive")
        if mode not in ("identity", "adaptive"):
            mode = "adaptive"
        cfg = MapperConfig(
            mode=mode,
            active_region=tuple(d.get("active_region", (0.0, 0.0, 1.0, 1.0))),
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


# ----------------------------------------------------------------------
# Demo: test básico
# ----------------------------------------------------------------------
if __name__ == "__main__":
    m = VirtualMapper(MapperConfig(mode="identity"))
    print("identity:", m.map_point(0.3, 0.7))
    m2 = VirtualMapper(MapperConfig(mode="adaptive"))
    print("adaptive (centro):", m2.map_point(0.5, 0.5))
    m2.update_from_hand_span(0.20)  # mano cerca
    print("adaptive cerca (centro):", m2.map_point(0.5, 0.5))
