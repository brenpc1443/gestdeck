"""
main.py — Orquestador del backend en Modo Virtual.

Inicia 4 hilos paralelos (adaptado al modo fantasma: no hay ArUco):
    Hilo 1 — Captura de cámara (interno de core.camera)
    Hilo 2 — Percepción MediaPipe + mapeo virtual + clasificación + broadcast
    Hilo 3 — WebSocket server (interno de core.websocket_server)
    Hilo 4 — Timers de calibración y tareas ligeras

Maneja los siguientes mensajes entrantes de Electron:
    reiniciar
    cambiar_slide
    iniciar_calibracion
    finalizar
    cargar_sesion
    guardar_sesion
    setup_session
    agregar_objeto
    mover_objeto
    eliminar_objeto
    escalar_objeto
    set_mirror
    set_mano_dominante

Uso:
    python -m core.main
    python -m core.main --config sessions/ejemplo_demo/config.json
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.camera import Camera, CameraConfig
from core.mediapipe_tracker import Tracker
from core.virtual_mapper import VirtualMapper, MapperConfig
from core.gesture_classifier import GestureClassifier
from core.object_engine import ObjectEngine, Objeto
from core.websocket_server import WebSocketServer
from core.training_manager import (
    TrainingManager,
    CLASES_ENTRENABLES_DOM,
    CLASES_ENTRENABLES_SUP,
    MODEL_ARCH_VERSION,
)


class GestDeckBackend:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self.session: dict = {}
        self.session_name: str = ""
        self.dominant: str = "derecha"
        self.total_slides: int = 1

        # Componentes
        self.camera = Camera(CameraConfig(mirror=True, apply_clahe=True))
        self.tracker = Tracker(process_every_n=2)
        # Modo adaptive: la región activa de la cámara se ajusta sola al
        # span aparente de la mano (proxy de distancia a la cámara). Eso
        # permite que la mano alcance los bordes del slide sin salirse del
        # encuadre, tanto presentando cerca como lejos. Si la sesión carga
        # una calibración manual (`load_session`), ese mapper la sobreescribe.
        self.mapper = VirtualMapper(MapperConfig(mode="adaptive"))
        self.engine = ObjectEngine()
        self.ws = WebSocketServer(
            on_message=self._on_message,
            on_connect=self._on_client_connect,
        )

        # Dos clasificadores LSTM independientes: uno por mano.
        # - Dominante: usa fallback rule-based para gestos estáticos.
        # - Apoyo: solo LSTM. Si no hay modelo entrenado, NINGUNO.
        self.classifier_dom = GestureClassifier(
            model_path="core/models/gestures_model_dominant.h5",
            labels_path="core/models/gestures_labels_dominant.json",
            use_rules=True,
            nombre="dominant",
        )
        self.classifier_sup = GestureClassifier(
            model_path="core/models/gestures_model_support.h5",
            labels_path="core/models/gestures_labels_support.json",
            use_rules=False,
            nombre="support",
        )
        # Dos training managers con clases y paths por mano.
        self.training_dom = TrainingManager(
            dataset_dir=f"training/dataset_inapp_v{MODEL_ARCH_VERSION}/dominant",
            model_path="core/models/gestures_model_dominant.h5",
            labels_path="core/models/gestures_labels_dominant.json",
            clases=CLASES_ENTRENABLES_DOM,
            nombre="dominant",
        )
        self.training_sup = TrainingManager(
            dataset_dir=f"training/dataset_inapp_v{MODEL_ARCH_VERSION}/support",
            model_path="core/models/gestures_model_support.h5",
            labels_path="core/models/gestures_labels_support.json",
            clases=CLASES_ENTRENABLES_SUP,
            nombre="support",
        )

        # Estado interno
        self._running = False
        self._perception_thread: Optional[threading.Thread] = None
        self._calibrating = False
        self._calibration_samples: list = []
        # Anti-rebote único para navegación de slides. La detección es por
        # FLANCO (cambio de gesto), pero un cooldown corto evita que una
        # oscilación SIGUIENTE↔NINGUNO del LSTM dispare dos veces seguidas.
        self._cooldown_nav = 0.0
        self._prev_nav_gesture: str = ""
        self._camera_ok = False
        self._camera_error: Optional[str] = None
        # Cuando está activo, _broadcast añade landmarks al payload para que
        # la pantalla de Entrenamiento dibuje ambas manos en vivo.
        self._training_mode = False
        # Últimos landmarks de cada mano (21, 3) — para save_sample y para
        # enviar al frontend durante entrenamiento.
        self._last_dominant_points = None
        self._last_support_points = None
        # Pausa por gesto de apoyo: toggle (cada vez que el gesto PAUSA
        # aparece por primera vez, alterna entre pausado y reanudado).
        self._paused = False
        self._prev_pause_gesture = False

    # ==================================================================
    # Ciclo de vida
    # ==================================================================
    def start(self) -> None:
        self._running = True
        self.ws.start()
        try:
            self.camera.start()
            self._camera_ok = True
            self._camera_error = None
        except Exception as e:
            self._camera_ok = False
            self._camera_error = str(e)
            print(f"[main] Cámara no disponible: {e}")
            # El backend sigue corriendo para que Electron reciba el aviso.
            self.ws.send({"tipo": "error_camara", "detalle": str(e)})
        # Anunciar estado de ambos clasificadores al conectarse cualquier cliente.
        self.ws.send({
            "tipo": "modo_gesto",
            "fuente": self.classifier_dom.source,
            "fuente_apoyo": self.classifier_sup.source,
        })
        if self._camera_ok:
            self.ws.send({"tipo": "camara_ok"})
        self._perception_thread = threading.Thread(
            target=self._perception_loop, daemon=True, name="gestdeck-perception")
        self._perception_thread.start()
        print(f"[main] Backend iniciado (cámara={'ok' if self._camera_ok else 'error'}, "
              f"dom={self.classifier_dom.source}, sup={self.classifier_sup.source}). "
              f"Ctrl+C para salir.")

    def stop(self) -> None:
        self._running = False
        if self._perception_thread is not None:
            self._perception_thread.join(timeout=1.0)
        self.camera.stop()
        self.tracker.close()
        self.ws.stop()
        print("[main] Backend detenido.")

    # ==================================================================
    # Gestión de sesiones
    # ==================================================================
    def load_session(self, path: Path) -> None:
        print(f"[main] Cargando sesión: {path}")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.session = data
        sesion_info = data.get("sesion", {})
        self.session_name = sesion_info.get("nombre", Path(path).parent.name)
        self.dominant = sesion_info.get("mano_dominante", "derecha")
        self.total_slides = int(sesion_info.get("total_slides", 1))

        objs = []
        for slide in data.get("slides", []):
            for o in slide.get("objetos", []):
                o.setdefault("slide_origen", slide["slide_id"])
                o.setdefault("slide_actual", slide["slide_id"])
                objs.append(Objeto.from_dict(o))
        self.engine.load(objs)

        cal = data.get("calibracion_virtual")
        if cal:
            self.mapper = VirtualMapper.from_dict(cal)

        self.config_path = Path(path).resolve()
        print(f"[main] {len(objs)} objetos, {self.total_slides} slides, "
              f"dominante={self.dominant}")

    def save_session(self, out_path: Optional[Path] = None) -> Path:
        """Serializa el estado actual a config.json."""
        dest = out_path or self.config_path
        if dest is None:
            raise ValueError("No hay ruta de sesión conocida. Indica una.")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        slides_data: dict[int, list] = {}
        for obj in self.engine.all():
            slide_id = obj.slide_origen
            slides_data.setdefault(slide_id, []).append(obj.to_dict())

        slides_out = []
        for sid in sorted(slides_data.keys() | {s["slide_id"] for s in self.session.get("slides", [])}):
            slides_out.append({
                "slide_id": sid,
                "objetos": slides_data.get(sid, []),
            })

        data = {
            "sesion": {
                "nombre": self.session_name or dest.parent.name,
                "fecha_guardado": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pptx_path": self.session.get("sesion", {}).get("pptx_path"),
                "slides_dir": self.session.get("sesion", {}).get("slides_dir"),
                "mano_dominante": self.dominant,
                "total_slides": self.total_slides,
                "modo": "ghost",
            },
            "calibracion_virtual": self.mapper.to_dict(),
            "slides": slides_out,
            "gestos_personalizados": self.session.get("gestos_personalizados", []),
        }
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[main] Sesión guardada: {dest}")
        return dest

    # ==================================================================
    # Bucle de percepción
    # ==================================================================
    def _perception_loop(self) -> None:
        last_t = time.time()
        camera_retry_at = 0.0
        while self._running:
            try:
                if not self._camera_ok:
                    # Intenta reabrir la cámara cada 3s sin tumbar el backend
                    now = time.time()
                    if now >= camera_retry_at:
                        try:
                            self.camera.start()
                            self._camera_ok = True
                            self._camera_error = None
                            print("[main] Cámara recuperada.")
                            self.ws.send({"tipo": "camara_ok"})
                        except Exception as e:
                            self._camera_error = str(e)
                            camera_retry_at = now + 3.0
                    time.sleep(0.2)
                    continue

                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                dt = now - last_t
                last_t = now

                tracking = self.tracker.process(frame)
                # for_user mapea (dominance, mirror) → slots por POSICIÓN x
                # del frame, ignorando la etiqueta L/R de MediaPipe (que se
                # invierte con mirror y causaba que la mano dominante física
                # fuera tratada como apoyo).
                dominant_hand, support_hand = tracking.for_user(
                    self.dominant, self.camera.config.mirror,
                )

                hand_xy: Optional[tuple] = None
                if dominant_hand is not None:
                    self.classifier_dom.push_frame(dominant_hand.flat_xy())
                    # Span aparente de la mano (wrist→MCP_medio en coords del
                    # frame): el mapper adaptive lo usa para inferir distancia
                    # a la cámara y ajustar la región activa.
                    pts = dominant_hand.points
                    span = float(np.linalg.norm(pts[9, :2] - pts[0, :2]))
                    self.mapper.update_from_hand_span(span)
                    palm = dominant_hand.palm_center
                    hand_xy = self.mapper.map_point(float(palm[0]), float(palm[1]))
                    self._last_dominant_points = pts
                    if self._calibrating:
                        self._calibration_samples.append((float(palm[0]), float(palm[1])))
                else:
                    self.classifier_dom.push_frame(None)
                    self._last_dominant_points = None

                if support_hand is not None:
                    self.classifier_sup.push_frame(support_hand.flat_xy())
                    self._last_support_points = support_hand.points
                else:
                    self.classifier_sup.push_frame(None)
                    self._last_support_points = None

                pred_dom = self.classifier_dom.predict(
                    dominant_hand.points if dominant_hand is not None else None
                )
                pred_sup = self.classifier_sup.predict(
                    support_hand.points if support_hand is not None else None
                )

                # --- Mano de apoyo: navegación y pausa -----------------
                # PAUSA es toggle por flanco: la primera vez que aparece
                # el gesto pausa; la siguiente vez reanuda. Soltar el
                # gesto no hace nada — sostenerlo tampoco dispara más.
                pause_now = (pred_sup.gesto == "PAUSA")
                if pause_now and not self._prev_pause_gesture:
                    self._paused = not self._paused
                    # Al entrar o salir de pausa, soltar el objeto activo
                    # para que el presentador "resetee" la interacción.
                    self.engine.deselect()
                self._prev_pause_gesture = pause_now

                # Navegación por FLANCO: solo dispara cuando el gesto aparece
                # (transición desde otro estado), no mientras se sostenga.
                # Sostener SIGUIENTE no debe avanzar slide tras slide.
                nav = pred_sup.gesto
                nav_edge = nav != self._prev_nav_gesture
                now_nav = time.time()
                if nav_edge and now_nav > self._cooldown_nav:
                    if nav == "SIGUIENTE":
                        self._cooldown_nav = now_nav + 0.5
                        self._go_to_slide(self.engine.current_slide + 1)
                    elif nav == "ANTERIOR":
                        self._cooldown_nav = now_nav + 0.5
                        self._go_to_slide(self.engine.current_slide - 1)
                self._prev_nav_gesture = nav

                # --- Mano dominante: objetos (bloqueado si PAUSA) ------
                if not self._paused:
                    self.engine.apply_gesture(
                        pred_dom.gesto, pred_dom.confianza, hand_xy, dt=dt
                    )
                else:
                    # Igual avanzamos la física (inercia, fricción) sin
                    # procesar el gesto — los objetos no se "congelan" en
                    # el aire con velocidad, sólo no responden a la mano.
                    self.engine.apply_gesture("NINGUNO", 0.0, hand_xy, dt=dt)

                self._broadcast(tracking, pred_dom, pred_sup, hand_xy)

                # Throttle ~60 fps
                time.sleep(max(0.0, 1 / 60 - (time.time() - now)))
            except Exception as e:
                # Safety net: una excepción puntual (MediaPipe, clasificador
                # en medio de un reload, etc.) no debe matar el hilo entero.
                # Si el hilo muriera, la cámara, landmarks y predicciones
                # quedarían congelados hasta reiniciar la app.
                print(f"[main] Error en perception loop (continuando): {e}")
                time.sleep(0.05)

    # ==================================================================
    # Broadcast
    # ==================================================================
    def _broadcast(self, tracking, pred_dom, pred_sup, hand_xy) -> None:
        active = self.engine.active_id
        active_obj = next((o for o in self.engine.all() if o.id == active), None)

        payload = {
            "tipo": "frame",
            # Campos del gesto DOMINANTE (compat con la UI existente).
            "gesto": pred_dom.gesto,
            "confianza": round(pred_dom.confianza, 3),
            "fuente": pred_dom.fuente,
            # Campos nuevos para la mano de APOYO y el estado de pausa.
            "gesto_apoyo": pred_sup.gesto,
            "confianza_apoyo": round(pred_sup.confianza, 3),
            "fuente_apoyo": pred_sup.fuente,
            "paused": self._paused,
            "mano_detectada": tracking.any_hand,
            "mano": {"x": hand_xy[0], "y": hand_xy[1]} if hand_xy else None,
            "objeto_activo": active,
            "estado_objeto": active_obj.estado if active_obj else None,
            "slide_actual": self.engine.current_slide,
            "total_slides": self.total_slides,
            "iluminacion_ok": True,
            "camera_fps": round(self.camera.fps, 1),
            "objetos": [o.to_dict() for o in self.engine.on_slide(self.engine.current_slide)],
        }
        if self._training_mode:
            # Landmarks de AMBAS manos para la pantalla de Entrenamiento.
            if self._last_dominant_points is not None:
                payload["landmarks_dominant"] = [
                    [float(p[0]), float(p[1])] for p in self._last_dominant_points
                ]
            if self._last_support_points is not None:
                payload["landmarks_support"] = [
                    [float(p[0]), float(p[1])] for p in self._last_support_points
                ]
            # Compat: algunos consumidores aún leen `landmarks` (dominante).
            if self._last_dominant_points is not None:
                payload["landmarks"] = payload["landmarks_dominant"]
        self.ws.send(payload)

    # ==================================================================
    # Navegación de slides
    # ==================================================================
    # La navegación por POSICIÓN de la mano de apoyo (borde del frame) fue
    # retirada en la Fase 1: ahora se dispara por GESTO LSTM de la apoyo
    # (SIGUIENTE / ANTERIOR). El cableado está directamente en el
    # perception loop.

    def _go_to_slide(self, new_slide: int) -> None:
        total = max(1, int(self.total_slides))
        new_slide = max(1, min(total, int(new_slide)))
        if new_slide == self.engine.current_slide:
            return
        carrying = self.engine.active_id is not None
        self.engine.change_slide(new_slide, carrying=carrying)
        self.ws.send({"tipo": "cambio_slide", "slide": new_slide})

    # ==================================================================
    # Estado inicial para un cliente que se conecta
    # ==================================================================
    def _on_client_connect(self) -> list:
        state: list = [{
            "tipo": "modo_gesto",
            "fuente": self.classifier_dom.source,
            "fuente_apoyo": self.classifier_sup.source,
        }]
        if self._camera_ok:
            state.append({"tipo": "camara_ok"})
        else:
            state.append({"tipo": "error_camara", "detalle": self._camera_error or "no disponible"})
        return state

    # ==================================================================
    # Mensajes entrantes desde Electron
    # ==================================================================
    def _on_message(self, msg: dict) -> None:
        tipo = msg.get("tipo")
        try:
            handler = getattr(self, f"_on_{tipo}", None)
            if handler is None:
                print(f"[main] Mensaje no reconocido: {tipo}")
                return
            handler(msg)
        except Exception as e:
            print(f"[main] Error procesando {tipo}: {e}")
            self.ws.send({"tipo": "error", "detalle": str(e), "mensaje_original": tipo})

    # --- handlers ------------------------------------------------------
    def _on_reiniciar(self, _msg):
        self.engine.reset_all()
        self.ws.send({"tipo": "reiniciado_ok"})

    def _on_cambiar_slide(self, msg):
        s = int(msg.get("slide", 1))
        self.engine.change_slide(s, carrying=False)
        self.ws.send({"tipo": "cambio_slide", "slide": s})

    def _on_iniciar_calibracion(self, _msg):
        self._calibrating = True
        self._calibration_samples.clear()
        threading.Timer(3.0, self._finish_calibration).start()
        self.ws.send({"tipo": "calibracion_en_curso"})

    def _on_finalizar(self, _msg):
        self.stop()

    def _on_cargar_sesion(self, msg):
        path = Path(msg["path"])
        if not path.exists():
            self.ws.send({"tipo": "error", "detalle": f"No existe {path}"})
            return
        self.load_session(path)
        self.ws.send({
            "tipo": "sesion_cargada",
            "slides": self.total_slides,
            "nombre": self.session_name,
            "dominante": self.dominant,
        })

    def _on_guardar_sesion(self, msg):
        path = msg.get("path")
        saved = self.save_session(Path(path) if path else None)
        self.ws.send({"tipo": "sesion_guardada", "path": str(saved)})

    def _on_setup_session(self, msg):
        """Usado por el Editor cuando se carga un PPTX nuevo desde cero."""
        self.session_name = msg.get("nombre", self.session_name or "nueva_sesion")
        self.total_slides = int(msg.get("total_slides", 1))
        self.engine.load([])
        meta = self.session.setdefault("sesion", {})
        meta["nombre"] = self.session_name
        meta["pptx_path"] = msg.get("pptx_path")
        meta["slides_dir"] = msg.get("slides_dir")
        meta["total_slides"] = self.total_slides
        self.session.setdefault("slides",
                                [{"slide_id": i + 1, "objetos": []} for i in range(self.total_slides)])
        self.ws.send({"tipo": "session_ready", "slides": self.total_slides})

    def _on_agregar_objeto(self, msg):
        slide = int(msg.get("slide", self.engine.current_slide))
        raw = dict(msg.get("objeto", {}))
        raw.setdefault("slide_origen", slide)
        raw.setdefault("slide_actual", slide)
        raw.setdefault("posicion_actual", raw.get("posicion_original", [0.5, 0.5]))
        raw.setdefault("escala", 1.0)
        raw.setdefault("rotacion", 0.0)
        raw.setdefault("estado", "EN_SLIDE")
        raw.setdefault("z_index", 1)
        obj = Objeto.from_dict(raw)
        self.engine.add(obj)
        self.ws.send({"tipo": "objeto_agregado", "objeto": obj.to_dict()})

    def _on_mover_objeto(self, msg):
        oid = msg["id"]
        x = float(msg["x"])
        y = float(msg["y"])
        update_origin = bool(msg.get("update_origin", True))
        ok = self.engine.move(oid, x, y, update_origin=update_origin)
        if not ok:
            self.ws.send({"tipo": "error", "detalle": f"Objeto {oid} no encontrado"})

    def _on_eliminar_objeto(self, msg):
        oid = msg["id"]
        self.engine.remove(oid)
        self.ws.send({"tipo": "objeto_eliminado", "id": oid})

    def _on_escalar_objeto(self, msg):
        oid = msg["id"]
        self.engine.scale(oid, float(msg.get("escala", 1.0)))

    def _on_set_mirror(self, msg):
        self.camera.config.mirror = bool(msg.get("value", True))

    def _on_set_mano_dominante(self, msg):
        v = str(msg.get("value", "derecha")).lower()
        if v.startswith("i") or v.startswith("l"):
            self.dominant = "izquierda"
        else:
            self.dominant = "derecha"
        self.ws.send({"tipo": "mano_dominante_set", "value": self.dominant})

    # -- Módulo de entrenamiento ---------------------------------------
    def _pair(self, mano: str):
        """Devuelve el par (classifier, training_manager) para la mano pedida.

        Acepta "dominant" / "dom" / "d"  →  dominante.
        Acepta "support"  / "sup" / "s"  →  apoyo.
        """
        m = (mano or "").strip().lower()
        if m.startswith("s"):
            return self.classifier_sup, self.training_sup
        return self.classifier_dom, self.training_dom

    def _build_stats(self) -> dict:
        """Stats completos: ambas manos + banderas de si se puede entrenar."""
        return {
            "dominant": {
                "stats": self.training_dom.stats(),
                "can_train": self.training_dom.can_train()[0],
            },
            "support": {
                "stats": self.training_sup.stats(),
                "can_train": self.training_sup.can_train()[0],
            },
        }

    def _on_training_start(self, _msg):
        """Activa el streaming de landmarks y envía el estado inicial por mano."""
        self._training_mode = True
        self.ws.send({
            "tipo": "training_ready",
            "clases_dominant": list(CLASES_ENTRENABLES_DOM),
            "clases_support": list(CLASES_ENTRENABLES_SUP),
            "stats": self._build_stats(),
            "fuente_dominant": self.classifier_dom.source,
            "fuente_support": self.classifier_sup.source,
        })

    def _on_training_stop(self, _msg):
        self._training_mode = False
        self.ws.send({"tipo": "training_stopped"})

    def _on_save_sample(self, msg):
        """Guarda la ventana actual como muestra. `mano` elige el dataset."""
        mano = str(msg.get("mano", "dominant"))
        clase = str(msg.get("clase", "")).strip().upper()
        clf, tm = self._pair(mano)
        seq = clf.get_current_sequence()
        if seq is None:
            self.ws.send({
                "tipo": "sample_error",
                "mano": mano,
                "detalle": "Aún no hay 30 frames capturados. Haz el gesto frente a la cámara y espera un segundo.",
            })
            return
        try:
            path = tm.save_sample(clase, seq)
        except ValueError as e:
            self.ws.send({"tipo": "sample_error", "mano": mano, "detalle": str(e)})
            return
        self.ws.send({
            "tipo": "sample_saved",
            "mano": mano,
            "clase": clase,
            "path": str(path),
            "stats": self._build_stats(),
        })

    def _on_dataset_stats(self, _msg):
        # Reenviamos también las clases entrenables — es la red de seguridad
        # por si el evento `training_ready` se perdió en el camino (race con
        # el setLastEvent de useWebSocket que sólo guarda el último evento).
        self.ws.send({
            "tipo": "dataset_stats",
            "stats": self._build_stats(),
            "clases_dominant": list(CLASES_ENTRENABLES_DOM),
            "clases_support": list(CLASES_ENTRENABLES_SUP),
        })

    def _on_retrain(self, msg):
        """Reentrena la mano especificada. `mano` = dominant|support.

        Si no se especifica, por defecto es la dominante.
        """
        mano = str(msg.get("mano", "dominant"))
        epochs = int(msg.get("epochs", 40))
        clf, tm = self._pair(mano)

        def progress(m: dict) -> None:
            m = {**m, "mano": mano}
            self.ws.send(m)
            if m.get("tipo") == "retrain_done" and m.get("modelo_ok"):
                ok = clf.reload_model()
                self.ws.send({
                    "tipo": "model_reloaded",
                    "mano": mano,
                    "modelo_ok": ok,
                    "fuente": clf.source,
                })
                self.ws.send({
                    "tipo": "modo_gesto",
                    "fuente": self.classifier_dom.source,
                    "fuente_apoyo": self.classifier_sup.source,
                })

        started = tm.train_async(on_progress=progress, epochs=epochs)
        if not started and tm.is_training:
            self.ws.send({
                "tipo": "retrain_error",
                "mano": mano,
                "detalle": "Ya hay un entrenamiento en curso.",
            })

    def _on_reload_model(self, msg):
        mano = str(msg.get("mano", "dominant"))
        clf, _ = self._pair(mano)
        ok = clf.reload_model()
        self.ws.send({
            "tipo": "model_reloaded",
            "mano": mano,
            "modelo_ok": ok,
            "fuente": clf.source,
        })
        self.ws.send({
            "tipo": "modo_gesto",
            "fuente": self.classifier_dom.source,
            "fuente_apoyo": self.classifier_sup.source,
        })

    # ==================================================================
    def _finish_calibration(self) -> None:
        self._calibrating = False
        if len(self._calibration_samples) >= 10:
            self.mapper.calibrate_from_samples(
                np.array(self._calibration_samples, dtype=np.float32))
            self.ws.send({"tipo": "calibracion_ok",
                          "active_region": list(self.mapper.config.active_region)})
        else:
            self.ws.send({"tipo": "calibracion_fallida"})


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GestDeck backend — Modo Virtual")
    parser.add_argument("--config", type=str, default=None,
                        help="Ruta a un config.json de sesión.")
    args = parser.parse_args()

    backend = GestDeckBackend(args.config)
    if args.config:
        backend.load_session(Path(args.config))

    def handle_sigint(_sig, _frame):
        backend.stop()
    signal.signal(signal.SIGINT, handle_sigint)

    backend.start()
    try:
        while backend._running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        backend.stop()


if __name__ == "__main__":
    main()
