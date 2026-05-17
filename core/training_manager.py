"""
training_manager.py — Orquestador del módulo de entrenamiento in-app.

Gestiona el dataset etiquetado por el usuario desde la pantalla de
Entrenamiento (validaciones, correcciones, grabaciones nuevas) y dispara
el reentrenamiento del LSTM sin bloquear el backend.

Layout en disco:

    training/dataset_inapp/
        PUNO_CERRADO/
            2026-04-20_14-32-10_123.npy    # (30, 42) float32
            ...
        PALMA_ABIERTA/
            ...

Cada archivo es UNA secuencia de 30 frames × 42 features (21 landmarks
aplanados x,y). Esta granularidad permite:
  - Contar muestras por clase sin parsear arrays grandes.
  - Borrar muestras individuales en el futuro si el usuario se equivoca.
  - Añadir muestras sin reescribir un .npy agregado.

El entrenamiento corre en un hilo daemon y emite progreso por un
callback que el WebSocket server reenvía a Electron.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Clases válidas por mano. Se separan porque cada clasificador (dominante
# y de apoyo) tiene su propio modelo LSTM y su propio dataset.
#
#   Dominante → interactúa con objetos (puño, palma, pellizco, índice,
#               empuje).
#   Apoyo     → controla la presentación (swipe siguiente/anterior,
#               pausa de reconocimiento).
CLASES_ENTRENABLES_DOM: List[str] = [
    "PUNO_CERRADO",
    "PALMA_ABIERTA",
    "PALMA_VERTICAL",
    "INDICE",
    "PELLIZCO",
    "PELLIZCO_INV",
    "EMPUJE",
]

CLASES_ENTRENABLES_SUP: List[str] = [
    "SIGUIENTE",
    "ANTERIOR",
    "PAUSA",
]

# Alias histórico — algún código antiguo podría importarlo.
CLASES_ENTRENABLES: List[str] = CLASES_ENTRENABLES_DOM + CLASES_ENTRENABLES_SUP

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 42
MIN_MUESTRAS_POR_CLASE = 5      # para permitir reentrenar
MIN_CLASES_CON_DATOS = 2

# Versión de la arquitectura + representación de features.
#
#   v1: LSTM(128)+LSTM(64) con Masking. Features absolutas.
#   v2: LSTM(64)+LSTM(32) sin Masking. Features absolutas.
#   v3: LSTM(64)+LSTM(32). Features = muñeca absoluta + 20 landmarks
#       relativos/escalados. Generaliza a distinta posición y escala.
#   v4: Dataset y modelos SEPARADOS por mano (dominante / apoyo). Mismo
#       tamaño de red por mano. Layout en disco:
#           training/dataset_inapp_v4/<mano>/<CLASE>/*.npy
#           core/models/gestures_model_<mano>.h5
#
# El classifier compara esta versión contra la escrita al lado del .h5
# y descarta modelos incompatibles. El directorio del dataset también
# lleva la versión para no mezclar muestras grabadas con distintas
# representaciones de features.
MODEL_ARCH_VERSION = 4


ProgressCallback = Callable[[dict], None]


class TrainingManager:
    """Gestiona dataset + entrenamiento de UNA mano. En el backend
    instanciamos uno para la mano dominante y otro para la mano de apoyo."""

    def __init__(
        self,
        dataset_dir: str | Path,
        model_path: str | Path,
        labels_path: str | Path,
        clases: List[str],
        nombre: str = "classifier",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.version_path = self.model_path.with_suffix(".version")
        self.clases: List[str] = list(clases)
        self.nombre = nombre
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._train_thread: Optional[threading.Thread] = None
        self._training: bool = False

    # ------------------------------------------------------------------
    # Gestión de muestras
    # ------------------------------------------------------------------
    def save_sample(self, clase: str, sequence: np.ndarray) -> Path:
        """Guarda una secuencia (30, 42) como muestra etiquetada.

        Lanza ValueError si la clase no es válida o la forma del array
        no es la esperada.
        """
        clase = clase.strip().upper()
        if clase not in self.clases:
            raise ValueError(f"Clase no válida: {clase}")
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
            raise ValueError(
                f"Shape inesperada {arr.shape}, "
                f"se esperaba ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME})"
            )
        clase_dir = self.dataset_dir / clase
        clase_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        ms = f"{int((time.time() % 1) * 1000):03d}"
        out = clase_dir / f"{stamp}_{ms}.npy"
        np.save(out, arr)
        return out

    def stats(self) -> Dict[str, int]:
        """Conteo de muestras por clase. Incluye las clases sin datos (0)."""
        counts: Dict[str, int] = {c: 0 for c in self.clases}
        for clase_dir in self.dataset_dir.iterdir() if self.dataset_dir.exists() else []:
            if not clase_dir.is_dir():
                continue
            nombre = clase_dir.name
            if nombre not in counts:
                counts[nombre] = 0
            counts[nombre] = sum(1 for p in clase_dir.glob("*.npy"))
        return counts

    def total_samples(self) -> int:
        return sum(self.stats().values())

    # ------------------------------------------------------------------
    # Carga para entrenamiento
    # ------------------------------------------------------------------
    def _load_all(self) -> tuple[np.ndarray, np.ndarray, List[str]]:
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        labels: List[str] = []
        for clase in sorted(self.clases):
            clase_dir = self.dataset_dir / clase
            if not clase_dir.exists():
                continue
            files = sorted(clase_dir.glob("*.npy"))
            # Una clase con menos del mínimo no entra al training: el split
            # estratificado se rompe y degrada al resto del modelo. Mejor
            # ignorarla por completo hasta que el usuario grabe suficientes.
            if len(files) < MIN_MUESTRAS_POR_CLASE:
                if files:
                    print(
                        f"[training:{self.nombre}] Ignorando clase '{clase}' "
                        f"({len(files)} muestras < mínimo {MIN_MUESTRAS_POR_CLASE})."
                    )
                continue
            label_idx = len(labels)
            labels.append(clase)
            for f in files:
                arr = np.load(f).astype(np.float32)
                if arr.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    continue
                X_list.append(arr)
                y_list.append(label_idx)
        if not X_list:
            raise RuntimeError("No hay muestras en el dataset in-app.")
        X = np.stack(X_list, axis=0)
        y = np.asarray(y_list, dtype=np.int64)
        return X, y, labels

    # ------------------------------------------------------------------
    # Validación previa al entreno
    # ------------------------------------------------------------------
    def can_train(self) -> tuple[bool, str]:
        counts = self.stats()
        con_datos = [c for c, n in counts.items() if n >= MIN_MUESTRAS_POR_CLASE]
        if len(con_datos) < MIN_CLASES_CON_DATOS:
            return (
                False,
                f"Se necesitan al menos {MIN_CLASES_CON_DATOS} clases con "
                f"{MIN_MUESTRAS_POR_CLASE}+ muestras cada una. "
                f"Ahora hay {len(con_datos)}.",
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # Entrenamiento asíncrono
    # ------------------------------------------------------------------
    @property
    def is_training(self) -> bool:
        return self._training

    def train_async(
        self,
        on_progress: ProgressCallback,
        epochs: int = 40,
        batch_size: int = 16,
    ) -> bool:
        """Arranca el entrenamiento en un hilo daemon.

        `on_progress` recibe dicts con alguna de estas formas:
          {"tipo": "retrain_started", "clases": [...], "muestras": N}
          {"tipo": "retrain_progress", "epoca": i, "total": E, "loss": f, "acc": f}
          {"tipo": "retrain_done", "modelo_ok": bool, "clases": [...]}
          {"tipo": "retrain_error", "detalle": str}

        Devuelve False si ya había un entrenamiento en curso.
        """
        if self._training:
            return False
        ok, reason = self.can_train()
        if not ok:
            on_progress({"tipo": "retrain_error", "detalle": reason})
            return False

        self._training = True
        self._train_thread = threading.Thread(
            target=self._train_worker,
            args=(on_progress, epochs, batch_size),
            daemon=True,
            name="gestdeck-retrain",
        )
        self._train_thread.start()
        return True

    def _train_worker(
        self,
        on_progress: ProgressCallback,
        epochs: int,
        batch_size: int,
    ) -> None:
        try:
            X, y, labels = self._load_all()
            on_progress({
                "tipo": "retrain_started",
                "clases": labels,
                "muestras": int(len(X)),
            })

            import tensorflow as tf  # noqa: F401 — import pesado diferido
            from tensorflow.keras import layers, Sequential
            from tensorflow.keras.callbacks import Callback, EarlyStopping

            # Libera estado acumulado (grafos, optimizadores) de sesiones
            # anteriores. Sin esto la inferencia se ralentiza reentreno tras
            # reentreno porque TF deja el modelo viejo en memoria.
            tf.keras.backend.clear_session()

            # --- Split estratificado: garantiza que cada clase aparezca
            # tanto en train como en val. Con datasets pequeños el split
            # random podía dejar clases enteras fuera de validación y
            # distorsionar las métricas.
            rng = np.random.default_rng(42)
            tr_idx: List[int] = []
            val_idx: List[int] = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                rng.shuffle(cls_idx)
                split = max(1, int(len(cls_idx) * 0.85))
                tr_idx.extend(cls_idx[:split].tolist())
                val_idx.extend(cls_idx[split:].tolist())
            rng.shuffle(tr_idx)
            rng.shuffle(val_idx)
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # --- Data augmentation sintética: triplica el dataset de train
            # con variantes plausibles de cada muestra. Combate el
            # overfitting sin pedirle al usuario grabar más.
            X_tr, y_tr = _augment(X_tr, y_tr, rng)

            # Arquitectura con regularización más fuerte para datasets
            # pequeños. Dropout sube a 0.4/0.5 — el modelo se hace más
            # "testarudo" durante el entrenamiento y generaliza mejor.
            model = Sequential([
                layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.4),
                layers.LSTM(32),
                layers.Dropout(0.4),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(len(labels), activation="softmax"),
            ])
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            total_epochs = int(epochs)

            class ProgressReporter(Callback):
                def on_epoch_end(self, epoca, logs=None):
                    logs = logs or {}
                    on_progress({
                        "tipo": "retrain_progress",
                        "epoca": int(epoca) + 1,
                        "total": total_epochs,
                        "loss": float(logs.get("loss", 0.0)),
                        "acc": float(logs.get("accuracy", 0.0)),
                        "val_loss": float(logs.get("val_loss", 0.0)),
                        "val_acc": float(logs.get("val_accuracy", 0.0)),
                    })

            # EarlyStopping: corta el entrenamiento si val_loss no mejora
            # en 8 épocas seguidas y restaura los pesos de la mejor época.
            # Evita seguir entrenando cuando ya solo se memoriza el train.
            callbacks = [ProgressReporter()]
            if len(X_val) > 0:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0,
                ))

            fit_kwargs = dict(
                epochs=total_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            if len(X_val) > 0:
                fit_kwargs["validation_data"] = (X_val, y_val)

            model.fit(X_tr, y_tr, **fit_kwargs)

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(self.model_path)
            self.labels_path.write_text(
                json.dumps(labels, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Escribe la versión de la arquitectura junto al modelo para
            # que el classifier rechace modelos entrenados con otra arq.
            self.version_path.write_text(str(MODEL_ARCH_VERSION), encoding="utf-8")

            on_progress({
                "tipo": "retrain_done",
                "modelo_ok": True,
                "clases": labels,
            })
        except Exception as e:
            on_progress({"tipo": "retrain_error", "detalle": str(e)})
        finally:
            self._training = False


# ----------------------------------------------------------------------
# Data augmentation
# ----------------------------------------------------------------------
def _augment(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    copies: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera variantes sintéticas plausibles de cada muestra.

    Cada muestra original produce `copies` variantes adicionales con
    perturbaciones pequeñas, triplicando (por defecto) el dataset de
    train. Las transformaciones están pensadas para que sigan siendo el
    mismo gesto en el mundo real:

      - jitter gaussiano en landmarks (σ pequeño): simula ruido de
        MediaPipe.
      - shift temporal ±1-2 frames: simula grabar un poco antes o
        después.
      - escalado multiplicativo en las features de forma: simula
        acercarse o alejarse ligeramente de la cámara.

    NO aplica volteo horizontal — el proyecto distingue izquierda y
    derecha, volver a etiquetar invertido confundiría al modelo.
    """
    if len(X) == 0 or copies <= 0:
        return X, y

    out_X = [X]
    out_y = [y]
    for _ in range(copies):
        Xc = X.copy()
        # 1) Jitter gaussiano, pequeño, sobre todas las features.
        Xc += rng.normal(0.0, 0.01, size=Xc.shape).astype(np.float32)
        # 2) Escalado ±5 % sobre la parte "forma" (índices 2-41), dejando
        #    la muñeca (0-1) intacta para no cambiar la trayectoria global.
        scale = rng.uniform(0.95, 1.05, size=(Xc.shape[0], 1, 1)).astype(np.float32)
        Xc[:, :, 2:] *= scale
        # 3) Shift temporal ±2 frames, rellenando el borde repetido.
        shifts = rng.integers(-2, 3, size=Xc.shape[0])
        for i, s in enumerate(shifts):
            if s == 0:
                continue
            if s > 0:
                Xc[i] = np.concatenate([np.repeat(Xc[i, :1], s, axis=0), Xc[i, :-s]], axis=0)
            else:
                Xc[i] = np.concatenate([Xc[i, -s:], np.repeat(Xc[i, -1:], -s, axis=0)], axis=0)
        out_X.append(Xc)
        out_y.append(y.copy())

    X_aug = np.concatenate(out_X, axis=0)
    y_aug = np.concatenate(out_y, axis=0)
    # Un shuffle final para que augmentations no queden agrupadas.
    perm = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tm = TrainingManager()
    print("Muestras actuales:", tm.stats())
    print("¿Se puede entrenar?:", tm.can_train())
