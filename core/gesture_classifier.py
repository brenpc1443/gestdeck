"""
gesture_classifier.py — Clasificación de gestos.

Dos modos disponibles:

1. LSTM (principal, Deep Learning)
   - Recibe una secuencia de N=30 frames de 42 features (21 landmarks x,y
     aplanados) y devuelve la clase de gesto + confianza.
   - Carga core/models/gestures_model.h5 entrenado por el propio usuario.

2. Rule-based (fallback)
   - Si no existe modelo entrenado, usa heurísticas geométricas simples
     sobre los landmarks del frame actual (puño, palma abierta, pellizco,
     índice, etc.). Esto permite usar la app antes de grabar gestos.

Nombres de clase exportados (en mayúsculas) siguen la Tabla 6.2 original:
    PUNO_CERRADO, PALMA_ABIERTA, INDICE, PELLIZCO, PELLIZCO_INV,
    PALMA_VERTICAL, EMPUJE, SIGUIENTE, ANTERIOR, NINGUNO
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np

# Constantes de la secuencia temporal
SEQUENCE_LENGTH = 30              # ~1 segundo a 30 fps procesados
FEATURES_PER_FRAME = 42           # 21 landmarks × (x, y)
CONFIDENCE_THRESHOLD = 0.5
# Debajo de este umbral el LSTM se considera "realmente perdido" y
# consultamos las reglas como rescate. Lo dejamos bajo a propósito para
# que el LSTM pueda "ganar" con confianzas moderadas (0.4-0.5) en gestos
# temporales (swipes, transiciones) que las reglas no saben detectar.
LSTM_RESCUE_THRESHOLD = 0.35
# El LSTM sólo se evalúa una de cada N llamadas a predict(). En los frames
# intermedios se reutiliza la predicción anterior. Con `tf.function` la
# inferencia es ya muy rápida, así que N=1 (cada frame) es seguro y da
# máxima agilidad. Subir a 2-3 si se detecta lag en CPUs modestas.
LSTM_PREDICT_EVERY_N = 1


@dataclass
class GesturePrediction:
    gesto: str
    confianza: float
    fuente: str                   # "lstm" o "rule-based"


class GestureClassifier:
    """Clasificador unificado con fallback a reglas."""

    def __init__(
        self,
        model_path: str | Path = "core/models/gestures_model.h5",
        labels_path: str | Path = "core/models/gestures_labels.json",
        use_rules: bool = True,
        nombre: str = "classifier",
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.use_rules = use_rules           # False para la mano de apoyo
        self.nombre = nombre
        self._sequence: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self._model = None
        self._predict_fn = None   # tf.function compilada con signature fija
        self._labels: List[str] = []
        self._last_pred: Optional[GesturePrediction] = None
        self._frame_counter: int = 0
        # Último frame válido: si MediaPipe pierde la mano un instante,
        # repetimos este frame en vez de inyectar ceros que ensuciarían
        # la ventana del LSTM (ahora que la capa Masking ya no existe).
        self._last_valid_frame: Optional[np.ndarray] = None
        self._load_model_if_available()

    # ------------------------------------------------------------------
    def _load_model_if_available(self) -> None:
        if not (self.model_path.exists() and self.labels_path.exists()):
            print(
                f"[gesture_classifier:{self.nombre}] Sin modelo entrenado "
                f"({self.model_path.name})."
            )
            return

        # Compara la versión de arquitectura del modelo guardado con la
        # actual del training_manager. Si no coincide o no hay archivo de
        # versión, ignora el .h5 (lo más probable es que sea de una arq
        # anterior y daría inferencia lenta o incorrecta).
        try:
            from core.training_manager import MODEL_ARCH_VERSION
        except Exception:
            MODEL_ARCH_VERSION = None
        version_path = self.model_path.with_suffix(".version")
        saved_version = None
        if version_path.exists():
            try:
                saved_version = int(version_path.read_text(encoding="utf-8").strip())
            except Exception:
                saved_version = None
        if MODEL_ARCH_VERSION is not None and saved_version != MODEL_ARCH_VERSION:
            print(
                f"[gesture_classifier:{self.nombre}] Modelo en disco es de "
                f"arquitectura v{saved_version} (esperada v{MODEL_ARCH_VERSION}). "
                f"Lo ignoro y uso rule-based hasta que reentrenes."
            )
            return

        try:
            import tensorflow as tf
            # Libera los grafos/pesos del modelo anterior antes de cargar
            # el nuevo. Sin esto, TF acumula estado entre reentrenamientos
            # y la inferencia se va degradando con cada ciclo.
            tf.keras.backend.clear_session()

            # IMPORTANTE: construimos todo en variables locales y recién al
            # final asignamos a self. El hilo de percepción usa `self._model`
            # como compuerta — si lo viera non-None con `_predict_fn` aún
            # a medio compilar, explotaría con 'NoneType not callable'.
            new_model = tf.keras.models.load_model(str(self.model_path))
            new_labels = json.loads(self.labels_path.read_text())

            # Compila la inferencia a un tf.function con signature fija.
            @tf.function(input_signature=[
                tf.TensorSpec(shape=(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME),
                              dtype=tf.float32)
            ])
            def _predict(x):
                return new_model(x, training=False)

            # Warmup con el propio _predict para dejar el grafo trazado.
            dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
            _ = _predict(tf.constant(dummy))

            # Commit atómico: el _predict_fn se publica ANTES que _model,
            # y _model es lo que abre la puerta en predict(). Así cuando
            # percepción vea _model ≠ None, _predict_fn ya está listo.
            self._labels = new_labels
            self._predict_fn = _predict
            self._model = new_model
            print(f"[gesture_classifier:{self.nombre}] LSTM cargada: {len(self._labels)} clases (v{saved_version}).")
        except Exception as e:
            print(f"[gesture_classifier:{self.nombre}] No se pudo cargar LSTM: {e}. Usando fallback.")
            self._model = None
            self._predict_fn = None

    @property
    def is_lstm(self) -> bool:
        return self._model is not None

    @property
    def source(self) -> str:
        return "lstm" if self._model is not None else "rule-based"

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def push_frame(self, flat_xy: np.ndarray) -> None:
        """Añade las 42 features (21 landmarks x,y) a la ventana deslizante.

        Si no hay mano detectada, repite el último frame válido en vez de
        meter ceros — eso mantiene la continuidad de la secuencia y evita
        que el LSTM vea un salto abrupto a (0,0,...) que desorientaría la
        clasificación. Si nunca hubo un frame válido aún, sí inyecta ceros
        como arranque.
        """
        if flat_xy is None or len(flat_xy) != FEATURES_PER_FRAME:
            if self._last_valid_frame is not None:
                flat_xy = self._last_valid_frame
            else:
                flat_xy = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
        else:
            flat_xy = flat_xy.astype(np.float32)
            self._last_valid_frame = flat_xy
        self._sequence.append(flat_xy)

    def get_current_sequence(self) -> Optional[np.ndarray]:
        """Devuelve la ventana actual de 30 frames × 42 features si está llena.

        Se usa desde el módulo de entrenamiento in-app para capturar la
        secuencia que está viendo el clasificador justo ahora y guardarla
        como muestra etiquetada.
        """
        if len(self._sequence) < SEQUENCE_LENGTH:
            return None
        return np.stack(list(self._sequence), axis=0).astype(np.float32)

    def reload_model(self) -> bool:
        """Recarga el .h5 desde disco (tras un reentrenamiento in-app).

        Devuelve True si tras la recarga hay modelo LSTM disponible.
        """
        # Cerramos la compuerta antes del reload para que predict() caiga
        # al rule-based durante la ventana en la que no hay modelo válido.
        self._model = None
        self._predict_fn = None
        self._labels = []
        self._load_model_if_available()
        return self._model is not None

    def predict(self, hand_points_xy: Optional[np.ndarray] = None) -> GesturePrediction:
        """Devuelve el gesto más probable.

        - Si hay modelo LSTM cargado y la secuencia está llena, usa la red.
        - Si no, usa la heurística geométrica sobre el frame actual
          (hand_points_xy debe tener shape (21, 2) o (21, 3)).
        """
        self._frame_counter += 1

        if (
            self._model is not None
            and self._predict_fn is not None
            and len(self._sequence) == SEQUENCE_LENGTH
        ):
            # Throttle: sólo corremos la red cada N frames. En los
            # intermedios reutilizamos la última predicción LSTM tal cual —
            # ya pasó por el filtro de confianza en su llamada original.
            is_lstm_turn = self._frame_counter % LSTM_PREDICT_EVERY_N == 0
            has_cached = (
                self._last_pred is not None
                and self._last_pred.fuente in ("lstm", "rule-based")
            )
            if is_lstm_turn or not has_cached:
                pred = self._predict_lstm()
                # Rescate por reglas sólo si están habilitadas (mano
                # dominante). Para la apoyo use_rules=False — no hay
                # reglas heurísticas de swipe/pausa, preferimos NINGUNO
                # a un falso positivo.
                if self.use_rules and pred.confianza < LSTM_RESCUE_THRESHOLD:
                    rule_pred = self._predict_rules(hand_points_xy)
                    if (
                        rule_pred.gesto != "NINGUNO"
                        and rule_pred.confianza > pred.confianza
                    ):
                        pred = rule_pred
            else:
                return self._last_pred
        elif self.use_rules:
            pred = self._predict_rules(hand_points_xy)
        else:
            # Apoyo sin modelo cargado: no inventar gestos por reglas.
            pred = GesturePrediction("NINGUNO", 0.0, "rule-based")

        # Filtro temporal: solo emitimos un cambio si la confianza supera el umbral.
        if pred.confianza < CONFIDENCE_THRESHOLD and self._last_pred is not None:
            pred = GesturePrediction(
                gesto="NINGUNO",
                confianza=pred.confianza,
                fuente=pred.fuente,
            )
        self._last_pred = pred
        return pred

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------
    def _predict_lstm(self) -> GesturePrediction:
        import tensorflow as tf
        seq = np.stack(list(self._sequence), axis=0)[np.newaxis, ...]   # (1, 30, 42)
        # Usa la tf.function compilada (mucho más rápida que model.predict()
        # o incluso que model(x, training=False) sin compilar).
        probs = self._predict_fn(tf.constant(seq, dtype=tf.float32)).numpy()[0]
        idx = int(np.argmax(probs))
        return GesturePrediction(
            gesto=self._labels[idx],
            confianza=float(probs[idx]),
            fuente="lstm",
        )

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    def _predict_rules(self, hp: Optional[np.ndarray]) -> GesturePrediction:
        if hp is None:
            return GesturePrediction("NINGUNO", 0.0, "rule-based")

        WRIST = 0
        THUMB_TIP, THUMB_MCP = 4, 2
        INDEX_TIP, INDEX_PIP = 8, 6
        MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18

        p = hp[:, :2]
        scale = np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]) + 1e-6

        def extended(tip: int, pip: int) -> bool:
            return np.linalg.norm(p[tip] - p[WRIST]) > np.linalg.norm(p[pip] - p[WRIST]) * 1.10

        index_out = extended(INDEX_TIP, INDEX_PIP)
        middle_out = extended(MIDDLE_TIP, MIDDLE_PIP)
        ring_out = extended(RING_TIP, RING_PIP)
        pinky_out = extended(PINKY_TIP, PINKY_PIP)
        thumb_out = (
            np.linalg.norm(p[THUMB_TIP] - p[WRIST])
            > np.linalg.norm(p[THUMB_MCP] - p[WRIST]) * 1.25
        )
        open_fingers = sum([index_out, middle_out, ring_out, pinky_out])

        pinch_dist = np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP]) / scale

        # Orientación palma: vector wrist → punta del medio.
        # Y crece hacia abajo en la imagen → palm_vec[1] < 0 = dedos hacia arriba.
        palm_vec = p[MIDDLE_TIP] - p[WRIST]
        palm_vertical = abs(palm_vec[1]) > abs(palm_vec[0]) * 1.3 and palm_vec[1] < 0

        last = self._last_pred.gesto if self._last_pred is not None else None

        # 1) Puño cerrado — sin dedos extendidos
        if open_fingers == 0:
            return GesturePrediction("PUNO_CERRADO", 0.85, "rule-based")

        # 2) Pellizco — pulgar e índice próximos (umbral relajado)
        if pinch_dist < 0.30 and open_fingers <= 2:
            return GesturePrediction("PELLIZCO", 0.78, "rule-based")

        # 3) Pellizco inverso — separación amplia con varios dedos abiertos
        if pinch_dist > 0.75 and open_fingers >= 3:
            return GesturePrediction("PELLIZCO_INV", 0.72, "rule-based")

        # 4) Índice solo
        if index_out and not middle_out and not ring_out and not pinky_out:
            return GesturePrediction("INDICE", 0.82, "rule-based")

        # 5) Mano abierta (4 dedos extendidos)
        if open_fingers == 4:
            # Transición puño → palma = gesto de soltar/empujar
            if last == "PUNO_CERRADO" and thumb_out:
                return GesturePrediction("EMPUJE", 0.72, "rule-based")
            if palm_vertical:
                return GesturePrediction("PALMA_VERTICAL", 0.75, "rule-based")
            return GesturePrediction("PALMA_ABIERTA", 0.80, "rule-based")

        return GesturePrediction("NINGUNO", 0.5, "rule-based")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    clf = GestureClassifier()
    # simular mano con todos los dedos extendidos
    dummy = np.zeros((21, 3), dtype=np.float32)
    dummy[:, 0] = np.linspace(0, 1, 21)
    print(clf.predict(dummy))
