"""
train_lstm.py — Entrena la red LSTM de clasificación de gestos.

Arquitectura (Deep Learning real — secuencia temporal):

    Input (30, 42)
      ↓
    LSTM(128, return_sequences=True) + Dropout(0.2)
      ↓
    LSTM(64)                            + Dropout(0.2)
      ↓
    Dense(64, relu)                     + Dropout(0.3)
      ↓
    Dense(num_classes, softmax)

Entrenamiento en CPU toma ~segundos con 5-10 gestos × 50 reps.
Salida:
    core/models/gestures_model.h5
    core/models/gestures_labels.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATASET_DIR = Path("training/dataset")
INDEX_FILE = DATASET_DIR / "gestures_index.json"
MODEL_OUT = Path("core/models/gestures_model.h5")
LABELS_OUT = Path("core/models/gestures_labels.json")


def load_dataset():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {INDEX_FILE}. Graba primero con record_gestures.py"
        )
    idx = json.loads(INDEX_FILE.read_text())
    X_list, y_list, labels = [], [], []
    for label_idx, (gesto, path) in enumerate(sorted(idx.items())):
        arr = np.load(path).astype(np.float32)        # (reps, 30, 42)
        X_list.append(arr)
        y_list.append(np.full(len(arr), label_idx, dtype=np.int64))
        labels.append(gesto)
        print(f"  {gesto:>14s}  reps={len(arr)}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, labels


def build_model(num_classes: int, seq_len: int = 30, feat: int = 42):
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential

    model = Sequential([
        layers.Input(shape=(seq_len, feat)),
        layers.Masking(mask_value=0.0),                 # ignora frames de ceros
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Cargando dataset...")
    X, y, labels = load_dataset()
    print(f"Dataset: X={X.shape} y={y.shape} labels={labels}")

    if len(labels) < 2:
        raise SystemExit(
            f"Se necesitan al menos 2 gestos distintos para entrenar "
            f"(actual: {len(labels)}).\n"
            "Graba más con: python -m training.record_gestures --gesto <NOMBRE>"
        )
    if len(X) < 10:
        raise SystemExit(
            f"Dataset demasiado pequeño ({len(X)} muestras). "
            "Graba al menos ~30 repeticiones por gesto."
        )

    # Shuffle + split 85/15
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = max(1, int(len(X) * 0.85))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = build_model(num_classes=len(labels))
    model.summary()

    # Si la partición deja val vacío (dataset mínimo), entrena sin validación
    fit_kwargs = dict(epochs=60, batch_size=16, verbose=2)
    if len(X_val) > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)

    print("\nEntrenando...")
    model.fit(X_tr, y_tr, **fit_kwargs)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    LABELS_OUT.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"\n✓ Modelo guardado en  {MODEL_OUT}")
    print(f"✓ Labels guardados en {LABELS_OUT}")


if __name__ == "__main__":
    main()
