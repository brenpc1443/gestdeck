"""
object_engine.py — Física y estado de los objetos del slide.

Gestiona el ciclo de vida de cada objeto (PNG, SVG, GIF, texto, figura)
y aplica la física simulada frame a frame.

Mapeo de gestos → acción (mano dominante):
  - INDICE:       selecciona (única vía). No mueve.
  - PELLIZCO:     arrastra (el objeto sigue a la mano).
  - PALMA_ABIERTA: suelta / deselecciona el activo.
  - PALMA_VERTICAL: flota frente al expositor.
  - PELLIZCO_INV: agranda (zoom in).
  - PUNO_CERRADO: achica (zoom out).
  - EMPUJE:       devuelve al origen con inercia.

Estados: EN_SLIDE, EN_MANO, FLOTANDO.

Las coordenadas están en [0, 1] × [0, 1] dentro del slide — el frontend
se encarga de escalarlas al tamaño real del canvas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Estados posibles de un objeto
EN_SLIDE = "EN_SLIDE"
EN_MANO = "EN_MANO"
FLOTANDO = "FLOTANDO"


@dataclass
class Objeto:
    id: str
    tipo: str                                      # png | svg | gif | text | shape
    archivo: str                                   # ruta relativa o data URL
    posicion_original: Tuple[float, float]
    posicion_actual: Tuple[float, float]
    escala: float = 1.0
    rotacion: float = 0.0                          # radianes
    slide_origen: int = 1
    slide_actual: int = 1
    estado: str = EN_SLIDE
    z_index: int = 0

    # Física
    velocidad: Tuple[float, float] = (0.0, 0.0)
    radio_interaccion: float = 0.08                # ~8% del slide

    # Parámetros visuales que el frontend respeta
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tipo": self.tipo,
            "archivo": self.archivo,
            "posicion_original": list(self.posicion_original),
            "posicion_actual": list(self.posicion_actual),
            "escala": self.escala,
            "rotacion": self.rotacion,
            "slide_origen": self.slide_origen,
            "slide_actual": self.slide_actual,
            "estado": self.estado,
            "z_index": self.z_index,
            "velocidad": list(self.velocidad),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Objeto":
        return cls(
            id=d["id"],
            tipo=d["tipo"],
            archivo=d["archivo"],
            posicion_original=tuple(d["posicion_original"]),
            posicion_actual=tuple(d.get("posicion_actual", d["posicion_original"])),
            escala=float(d.get("escala", 1.0)),
            rotacion=float(d.get("rotacion", 0.0)),
            slide_origen=int(d.get("slide_origen", 1)),
            slide_actual=int(d.get("slide_actual", d.get("slide_origen", 1))),
            estado=d.get("estado", EN_SLIDE),
            z_index=int(d.get("z_index", 0)),
            meta=d.get("meta", {}),
        )


class ObjectEngine:
    """Gestiona todos los objetos y aplica la física frame a frame."""

    def __init__(
        self,
        friction: float = 0.85,
        magnet_strength: float = 0.15,
        grab_radius: float = 0.10,
        select_radius: float = 0.15,
        release_cooldown: float = 0.35,
    ):
        self._objects: Dict[str, Objeto] = {}
        self._active_id: Optional[str] = None
        self._current_slide: int = 1
        self.friction = friction
        self.magnet_strength = magnet_strength
        self.grab_radius = grab_radius
        # select_radius es más laxo que grab_radius porque "señalar" admite
        # menos precisión que "agarrar". El drag (PELLIZCO/etc.) sigue usando
        # grab_radius * 2.5 para auto-soltar — eso no cambia.
        self.select_radius = select_radius
        # Tras soltar un objeto, no permitir que INDICE lo re-seleccione
        # durante este intervalo. Resuelve el caso "arrastré A cerca de B,
        # suelto, INDICE re-agarra A en vez de B".
        self.release_cooldown = release_cooldown
        self._recent_release_id: Optional[str] = None
        self._recent_release_t: float = 0.0

    # ------------------------------------------------------------------
    # gestión de objetos
    # ------------------------------------------------------------------
    def load(self, objetos: List[Objeto]) -> None:
        self._objects = {o.id: o for o in objetos}

    def add(self, obj: Objeto) -> None:
        self._objects[obj.id] = obj

    def remove(self, obj_id: str) -> None:
        self._objects.pop(obj_id, None)
        if self._active_id == obj_id:
            self._active_id = None

    def move(self, obj_id: str, x: float, y: float, update_origin: bool = True) -> bool:
        """Mueve un objeto por drag manual desde el editor.

        update_origin=True actualiza también la posición ORIGEN (útil cuando
        el presentador está configurando la sesión), de modo que el botón
        "Reset" devuelva al objeto al nuevo sitio.
        """
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        obj.posicion_actual = (float(x), float(y))
        if update_origin:
            obj.posicion_original = (float(x), float(y))
        obj.velocidad = (0.0, 0.0)
        obj.estado = EN_SLIDE
        return True

    def scale(self, obj_id: str, escala: float) -> bool:
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        obj.escala = max(0.1, min(5.0, float(escala)))
        return True

    def all(self) -> List[Objeto]:
        return list(self._objects.values())

    def on_slide(self, slide: int) -> List[Objeto]:
        return [o for o in self._objects.values() if o.slide_actual == slide]

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    @property
    def current_slide(self) -> int:
        return self._current_slide

    # ------------------------------------------------------------------
    # interacciones basadas en gesto
    # ------------------------------------------------------------------
    def apply_gesture(
        self,
        gesto: str,
        confianza: float,
        hand_xy: Optional[Tuple[float, float]],
        dt: float = 1.0 / 30.0,
    ) -> None:
        """Aplica un gesto reconocido al motor de objetos."""
        self._step_physics(dt)

        if hand_xy is None:
            self._release_if_needed()
            return

        if gesto == "INDICE":
            # ÚNICA vía de selección: señalar el objeto con el índice. El
            # resto de gestos no pueden "adoptar" un objeto por proximidad.
            #
            # Excluimos el objeto recién soltado durante release_cooldown.
            # Sin esto, si el usuario arrastró A cerca de B y soltó, el
            # INDICE siguiente se quedaría con A (más cercano) en vez de B.
            now = time.time()
            exclude = (
                self._recent_release_id
                if (now - self._recent_release_t) < self.release_cooldown
                else None
            )
            obj = self._closest_on_slide(hand_xy, exclude_id=exclude)
            if obj is not None and self._distance(obj.posicion_actual, hand_xy) < self.select_radius:
                self._active_id = obj.id

        elif gesto == "PELLIZCO":
            # Arrastrar: el objeto activo sigue a la mano (pinzar y mover,
            # como en touch/AR). Sólo si hay objeto seleccionado con INDICE.
            # speed=1.0 → snap directo a la mano (sin smoothing del backend);
            # el frontend ya hace lerp suave a 60Hz, así que aquí queremos
            # entregar el target lo más fresco posible.
            obj = self._get_active_if_nearby(hand_xy)
            if obj is not None:
                obj.estado = EN_MANO
                self._move_towards(obj, hand_xy, speed=1.0)

        elif gesto == "PALMA_ABIERTA":
            # Soltar / liberar: termina la interacción con el objeto activo.
            # El objeto se queda donde está (EN_SLIDE) y se deselecciona.
            self.deselect()

        elif gesto == "PALMA_VERTICAL":
            # Flota frente al expositor.
            obj = self._get_active_if_nearby(hand_xy)
            if obj is not None:
                obj.estado = FLOTANDO
                self._move_towards(obj, hand_xy, speed=0.15)

        elif gesto == "PELLIZCO_INV":
            # Dedos separándose → expande el objeto (zoom in).
            obj = self._get_active_if_nearby(hand_xy)
            if obj is not None:
                obj.escala = min(3.0, obj.escala * 1.02)

        elif gesto == "PUNO_CERRADO":
            # Puño cerrado → comprime el objeto (zoom out).
            obj = self._get_active_if_nearby(hand_xy)
            if obj is not None:
                obj.escala = max(0.3, obj.escala * 0.98)

        elif gesto == "EMPUJE":
            # devolver al slide con velocidad hacia origen
            self._release_to_origin()

        elif gesto in ("SIGUIENTE", "ANTERIOR"):
            # navegación — el cambio de slide lo maneja el orquestador
            pass

        else:
            # NINGUNO / desconocido — no hacer nada sobre los objetos.
            # Sólo deseleccionamos si la mano se alejó del activo, pero
            # no atraemos por magnetismo: el usuario pasar la mano cerca
            # de un objeto no debe agarrarlo sin querer.
            if self._active_id and self._active_id in self._objects:
                obj = self._objects[self._active_id]
                if self._distance(obj.posicion_actual, hand_xy) > self.grab_radius * 2.5:
                    if obj.estado in (EN_MANO, FLOTANDO):
                        obj.estado = EN_SLIDE
                    self._mark_released()

    # ------------------------------------------------------------------
    # navegación de slides
    # ------------------------------------------------------------------
    def change_slide(self, nuevo: int, carrying: bool) -> None:
        """Al cambiar de slide, mueve el objeto activo con el presentador
        si carrying=True (cualquier objeto seleccionado se lleva al nuevo slide)."""
        if carrying and self._active_id is not None:
            obj = self._objects[self._active_id]
            obj.slide_actual = nuevo
        self._current_slide = nuevo
        # Los demás objetos permanecen en su slide de origen

    # ------------------------------------------------------------------
    # física
    # ------------------------------------------------------------------
    def _step_physics(self, dt: float) -> None:
        for obj in self._objects.values():
            if obj.estado in (EN_MANO, FLOTANDO):
                # controlado por la mano: la velocidad la setea _move_towards
                continue
            vx, vy = obj.velocidad
            x, y = obj.posicion_actual
            x += vx * dt
            y += vy * dt
            # fricción
            vx *= self.friction
            vy *= self.friction
            if abs(vx) < 1e-4 and abs(vy) < 1e-4:
                vx = vy = 0.0
            obj.posicion_actual = (x, y)
            obj.velocidad = (vx, vy)

    def _move_towards(self, obj: Objeto, target: Tuple[float, float], speed: float) -> None:
        cx, cy = obj.posicion_actual
        tx, ty = target
        nx = cx + (tx - cx) * speed
        ny = cy + (ty - cy) * speed
        obj.velocidad = ((nx - cx) * 30.0, (ny - cy) * 30.0)   # guardada para la inercia
        obj.posicion_actual = (nx, ny)

    def _apply_magnet(self, hand_xy: Tuple[float, float]) -> None:
        for obj in self._objects.values():
            if obj.slide_actual != self._current_slide:
                continue
            if obj.estado != EN_SLIDE:
                continue
            d = self._distance(obj.posicion_actual, hand_xy)
            if d < 0.18:
                pull = self.magnet_strength * (1.0 - d / 0.18)
                cx, cy = obj.posicion_actual
                tx, ty = hand_xy
                obj.posicion_actual = (
                    cx + (tx - cx) * pull,
                    cy + (ty - cy) * pull,
                )

    def _release_if_needed(self) -> None:
        """Sin mano visible: soltar el objeto activo en EN_SLIDE y liberar el slot."""
        if self._active_id and self._active_id in self._objects:
            obj = self._objects[self._active_id]
            if obj.estado in (EN_MANO, FLOTANDO):
                obj.estado = EN_SLIDE
                obj.velocidad = (0.0, 0.0)
            self._mark_released()

    def _release_to_origin(self) -> None:
        if self._active_id is None:
            return
        obj = self._objects[self._active_id]
        # impulso hacia la posición original
        ox, oy = obj.posicion_original
        cx, cy = obj.posicion_actual
        obj.velocidad = ((ox - cx) * 4.0, (oy - cy) * 4.0)
        obj.estado = EN_SLIDE
        self._mark_released()

    def _mark_released(self) -> None:
        """Limpia el activo y arma el cooldown anti-reseleccion para ese id.

        Cualquier ruta de "soltar" debe pasar por aquí: PALMA_ABIERTA,
        EMPUJE, auto-release por distancia, toggle de pausa. Así el
        siguiente INDICE no podrá re-agarrar el mismo objeto durante
        release_cooldown segundos.
        """
        if self._active_id is not None:
            self._recent_release_id = self._active_id
            self._recent_release_t = time.time()
        self._active_id = None

    def deselect(self) -> None:
        """Suelta el objeto activo en su posición actual (PALMA_ABIERTA, toggle de pausa)."""
        if not self._active_id or self._active_id not in self._objects:
            self._active_id = None
            return
        obj = self._objects[self._active_id]
        if obj.estado in (EN_MANO, FLOTANDO):
            obj.estado = EN_SLIDE
        obj.velocidad = (0.0, 0.0)
        self._mark_released()

    def reset_all(self) -> None:
        """Vuelve todos los objetos a su posición original (botón R)."""
        for obj in self._objects.values():
            obj.posicion_actual = obj.posicion_original
            obj.slide_actual = obj.slide_origen
            obj.escala = 1.0
            obj.rotacion = 0.0
            obj.estado = EN_SLIDE
            obj.velocidad = (0.0, 0.0)
        self._active_id = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _closest_on_slide(
        self,
        xy: Tuple[float, float],
        exclude_id: Optional[str] = None,
    ) -> Optional[Objeto]:
        objs = self.on_slide(self._current_slide)
        if exclude_id is not None:
            objs = [o for o in objs if o.id != exclude_id]
        if not objs:
            return None
        # Distancia primero (criterio natural al "señalar"); el z_index sólo
        # rompe empates cuando dos objetos están casi superpuestos. Antes el
        # sort era (-z_index, distancia) y un objeto con z_index alto podía
        # ganar aunque la mano estuviera lejos.
        objs.sort(key=lambda o: (self._distance(o.posicion_actual, xy), -o.z_index))
        return objs[0]

    def _get_active_if_nearby(self, xy: Tuple[float, float]) -> Optional[Objeto]:
        """Devuelve el objeto activo SI y solo si existe y la mano sigue cerca.

        NUNCA busca el más cercano por proximidad: la selección es responsabilidad
        exclusiva del gesto INDICE. Así puño/palma/pellizco jamás "adoptan" un
        objeto que el usuario no haya señalado antes — evita interacciones
        accidentales mientras gesticula al exponer.
        """
        if not self._active_id or self._active_id not in self._objects:
            return None
        obj = self._objects[self._active_id]
        if self._distance(obj.posicion_actual, xy) >= self.grab_radius * 2.5:
            # Mano se alejó demasiado → liberar y volver a EN_SLIDE.
            if obj.estado in (EN_MANO, FLOTANDO):
                obj.estado = EN_SLIDE
            self._mark_released()
            return None
        return obj


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    eng = ObjectEngine()
    eng.load([
        Objeto(id="a", tipo="svg", archivo="a.svg",
               posicion_original=(0.3, 0.5), posicion_actual=(0.3, 0.5), slide_origen=1),
    ])
    eng.apply_gesture("INDICE", 0.9, (0.3, 0.5))
    eng.apply_gesture("PELLIZCO", 0.9, (0.6, 0.6))
    print(eng.all()[0].to_dict())
