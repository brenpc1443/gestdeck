"""
record_gestures.py — Graba ejemplos de un gesto con la cámara.

Cada gesto se graba como N repeticiones. Cada repetición es una secuencia
de SEQUENCE_LENGTH=30 frames de 42 features (21 landmarks * (x, y)).

Salida:
    training/dataset/<NOMBRE_GESTO>.npy   — array (reps, 30, 42) float32
    training/dataset/gestures_index.json  — diccionario gesto→ruta

Uso:
    python -m training.record_gestures --gesto AGARRAR --repeticiones 50
    python -m training.record_gestures --gesto SOLTAR  --repeticiones 50

Flujo de grabación:
    1. La app abre la cámara y muestra el feed.
    2. Cuando pulses ESPACIO, empieza una repetición (cuenta regresiva 1s).
    3. Durante 1 segundo graba 30 frames; luego descansa 0.5s.
    4. Repite hasta completar --repeticiones.
    5. Pulsa Q para abortar.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.camera import Camera, CameraConfig
from core.mediapipe_tracker import Tracker

SEQUENCE_LENGTH = 30
FEATURES = 42
DATASET_DIR = Path("training/dataset")
INDEX_FILE = DATASET_DIR / "gestures_index.json"


def record(gesto: str, repeticiones: int, hand: str = "right") -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    samples: List[np.ndarray] = []

    with Camera(CameraConfig(mirror=True)) as cam, Tracker(process_every_n=1) as tracker:
        print(f"\n=== Grabando gesto: {gesto} ({repeticiones} repeticiones) ===")
        print("Pulsa ESPACIO para grabar una repetición. Q para salir.\n")
        rep_idx = 0
        recording = False
        seq: List[np.ndarray] = []
        countdown_until = 0.0

        while rep_idx < repeticiones:
            frame = cam.read()
            if frame is None:
                continue

            res = tracker.process(frame)
            hand_obj = res.right_hand if hand == "right" else res.left_hand
            vis = frame.copy()
            h, w = vis.shape[:2]

            if hand_obj is not None:
                for (x, y, _) in hand_obj.points:
                    cv2.circle(vis, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

            # Texto HUD
            cv2.putText(vis, f"Gesto: {gesto}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis, f"Reps: {rep_idx}/{repeticiones}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.time()
            if recording:
                if now < countdown_until:
                    remaining = int(countdown_until - now) + 1
                    cv2.putText(vis, f"Listo en {remaining}", (w // 2 - 100, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    # Grabando
                    cv2.rectangle(vis, (5, 5), (w - 5, h - 5), (0, 0, 255), 4)
                    if hand_obj is not None:
                        seq.append(hand_obj.flat_xy())
                    else:
                        seq.append(np.zeros(FEATURES, dtype=np.float32))
                    cv2.putText(vis, f"GRABANDO ({len(seq)}/{SEQUENCE_LENGTH})",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if len(seq) >= SEQUENCE_LENGTH:
                        samples.append(np.stack(seq, axis=0))
                        seq = []
                        recording = False
                        rep_idx += 1
                        print(f"  ✓ Repetición {rep_idx}/{repeticiones}")
                        time.sleep(0.5)
            else:
                cv2.putText(vis, "ESPACIO: grabar rep", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.imshow("GestDeck — record_gestures", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not recording:
                recording = True
                countdown_until = now + 1.0
                seq = []

    cv2.destroyAllWindows()

    if not samples:
        print("No se grabaron repeticiones. Abortando.")
        return

    arr = np.stack(samples, axis=0)
    out = DATASET_DIR / f"{gesto}.npy"
    np.save(out, arr)
    print(f"Guardadas {len(samples)} reps en {out}  shape={arr.shape}")

    # Actualiza el índice
    idx = {}
    if INDEX_FILE.exists():
        idx = json.loads(INDEX_FILE.read_text())
    idx[gesto] = str(out).replace("\\", "/")
    INDEX_FILE.write_text(json.dumps(idx, indent=2, ensure_ascii=False))
    print(f"Índice actualizado: {INDEX_FILE}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gesto", required=True, help="Nombre del gesto (ej. AGARRAR)")
    ap.add_argument("--repeticiones", type=int, default=50)
    ap.add_argument("--mano", choices=["right", "left"], default="right")
    args = ap.parse_args()
    record(args.gesto.strip().upper(), args.repeticiones, args.mano)


if __name__ == "__main__":
    main()
