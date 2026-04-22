// Training.jsx — Pantalla de entrenamiento / validación de gestos.
//
// Flujo (todo con botones visuales, sin atajos de teclado):
//   1. Hace un gesto frente a la cámara → ve la mano en vivo (21 landmarks)
//      y la predicción actual (fuente + confianza).
//   2. Botón ✓ CORRECTO → guarda la ventana actual (30 frames) como
//      muestra etiquetada con la clase predicha.
//   3. Botón ✗ NO ES ESE → abre el selector de clases (tarjetas grandes).
//      Al elegir la correcta, guarda la misma ventana con esa etiqueta.
//   4. Botón ● GRABAR → cuenta regresiva 3-2-1, graba 1 seg, abre selector
//      de clases para etiquetar.
//   5. Botón 🔁 REENTRENAR → dispara el LSTM en el backend y muestra
//      progreso (época, loss, accuracy) en un modal. Al terminar, el
//      modelo se recarga solo y la fuente pasa de "rule-based" a "lstm".

import React, { useEffect, useRef, useState } from 'react';
import useWebSocket from '../hooks/useWebSocket.js';

// Conexiones MediaPipe Hand para dibujar los huesos entre landmarks.
const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],              // pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],              // índice
  [5, 9], [9, 10], [10, 11], [11, 12],         // medio
  [9, 13], [13, 14], [14, 15], [15, 16],       // anular
  [13, 17], [17, 18], [18, 19], [19, 20],      // meñique
  [0, 17],                                     // palma
];

const CLASS_ICONS = {
  PUNO_CERRADO: '✊',
  PALMA_ABIERTA: '✋',
  PALMA_VERTICAL: '🖐️',
  INDICE: '☝️',
  PELLIZCO: '🤏',
  PELLIZCO_INV: '🫴',
  EMPUJE: '🫸',
  SIGUIENTE: '👉',
  ANTERIOR: '👈',
  PAUSA: '⏸️',
};

const CLASS_LABELS = {
  PUNO_CERRADO: 'Puño cerrado',
  PALMA_ABIERTA: 'Palma abierta',
  PALMA_VERTICAL: 'Palma vertical',
  INDICE: 'Índice',
  PELLIZCO: 'Pellizco',
  PELLIZCO_INV: 'Pellizco inverso',
  EMPUJE: 'Empuje',
  SIGUIENTE: 'Siguiente',
  ANTERIOR: 'Anterior',
  PAUSA: 'Pausa',
};

export default function Training() {
  const { connected, lastFrame, lastEvent, send } = useWebSocket();

  // Mano seleccionada: 'dominant' o 'support'. Todo lo que hagas aquí
  // (validar, grabar, reentrenar) aplica a ESTA mano.
  const [mano, setMano] = useState('dominant');
  // Clases y stats separadas por mano.
  const [clasesPorMano, setClasesPorMano] = useState({ dominant: [], support: [] });
  const [statsPorMano, setStatsPorMano] = useState({
    dominant: { stats: {}, can_train: false },
    support:  { stats: {}, can_train: false },
  });
  const [sourceMode, setSourceMode] = useState({ dominant: null, support: null });
  const [toast, setToast] = useState(null);
  const [picker, setPicker] = useState(null);        // { mode: 'correct'|'record' }
  const [recording, setRecording] = useState(null);  // { phase:'count'|'rec', ... }
  const [retrain, setRetrain] = useState(null);      // { mano, running, epoca, ... }

  // Al entrar: avisar al backend que active streaming de landmarks.
  useEffect(() => {
    if (!connected) return;
    send({ tipo: 'training_start' });
    send({ tipo: 'dataset_stats' });
    return () => send({ tipo: 'training_stop' });
  }, [connected, send]);

  // Eventos del backend.
  useEffect(() => {
    if (!lastEvent) return;
    switch (lastEvent.tipo) {
      case 'training_ready':
        setClasesPorMano({
          dominant: lastEvent.clases_dominant || [],
          support:  lastEvent.clases_support  || [],
        });
        if (lastEvent.stats) setStatsPorMano(lastEvent.stats);
        setSourceMode({
          dominant: lastEvent.fuente_dominant || null,
          support:  lastEvent.fuente_support  || null,
        });
        break;
      case 'dataset_stats':
        if (lastEvent.stats) setStatsPorMano(lastEvent.stats);
        break;
      case 'sample_saved':
        if (lastEvent.stats) setStatsPorMano(lastEvent.stats);
        setToast({
          text: `✓ Muestra guardada en ${lastEvent.clase} (${lastEvent.mano === 'support' ? 'apoyo' : 'dominante'})`,
          kind: 'ok',
        });
        break;
      case 'sample_error':
        setToast({ text: `⚠ ${lastEvent.detalle}`, kind: 'err' });
        break;
      case 'modo_gesto':
        setSourceMode({
          dominant: lastEvent.fuente || null,
          support:  lastEvent.fuente_apoyo || null,
        });
        break;
      case 'retrain_started':
        setRetrain({
          running: true, epoca: 0, total: 0,
          loss: 0, acc: 0, error: null, done: false,
          clases: lastEvent.clases, muestras: lastEvent.muestras,
        });
        break;
      case 'retrain_progress':
        setRetrain((r) => {
          const prev = r || { running: true };
          // Rastreamos la mejor época (val_loss mínimo) en vivo. Al terminar
          // con early stopping, Keras restaura los pesos de esa época aunque
          // siga reportando progreso algunas épocas más.
          const newBest =
            prev.bestValLoss == null || lastEvent.val_loss < prev.bestValLoss;
          return {
            ...prev,
            running: true,
            epoca: lastEvent.epoca,
            total: lastEvent.total,
            loss: lastEvent.loss,
            acc: lastEvent.acc,
            val_loss: lastEvent.val_loss,
            val_acc: lastEvent.val_acc,
            bestValLoss: newBest ? lastEvent.val_loss : prev.bestValLoss,
            bestValAcc: newBest ? lastEvent.val_acc : prev.bestValAcc,
            bestEpoch: newBest ? lastEvent.epoca : prev.bestEpoch,
          };
        });
        break;
      case 'retrain_done':
        setRetrain((r) => {
          const prev = r || {};
          const stoppedEarly =
            prev.epoca != null && prev.total != null && prev.epoca < prev.total;
          return {
            ...prev,
            running: false,
            done: true,
            stoppedEarly,
          };
        });
        setToast({ text: '✓ Modelo reentrenado y recargado', kind: 'ok' });
        break;
      case 'retrain_error':
        setRetrain((r) => ({ ...(r || {}), running: false, error: lastEvent.detalle }));
        break;
      default:
        break;
    }
  }, [lastEvent]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2500);
    return () => clearTimeout(t);
  }, [toast]);

  // --- Acciones (todo opera sobre la MANO seleccionada) ---------------
  // Gesto y landmarks según la mano activa.
  const landmarksDom = lastFrame?.landmarks_dominant || null;
  const landmarksSup = lastFrame?.landmarks_support || null;
  const landmarksActivos = mano === 'support' ? landmarksSup : landmarksDom;
  const gestoActual = mano === 'support' ? lastFrame?.gesto_apoyo : lastFrame?.gesto;
  const confianza   = mano === 'support' ? lastFrame?.confianza_apoyo : lastFrame?.confianza;
  const fuenteActual = mano === 'support' ? lastFrame?.fuente_apoyo : lastFrame?.fuente;
  const manoVisible = !!landmarksActivos;

  const clases = clasesPorMano[mano] || [];
  const stats  = statsPorMano[mano]?.stats || {};
  const canTrainMano = statsPorMano[mano]?.can_train || false;

  const handleCorrect = () => {
    if (!gestoActual || gestoActual === 'NINGUNO') {
      setToast({ text: 'Haz un gesto antes de validarlo', kind: 'err' });
      return;
    }
    send({ tipo: 'save_sample', mano, clase: gestoActual });
  };

  const handleWrong = () => setPicker({ mode: 'correct' });

  const handleRecord = () => {
    // Cuenta regresiva 3-2-1 visual, luego graba 1 segundo con barra de
    // progreso animada y abre el picker al terminar.
    setRecording({ phase: 'count', seconds: 3 });
    const tick = (n) => {
      if (n > 0) {
        setRecording({ phase: 'count', seconds: n });
        setTimeout(() => tick(n - 1), 1000);
      } else {
        const RECORD_MS = 1000;
        const start = performance.now();
        setRecording({ phase: 'rec', progress: 0 });
        const raf = () => {
          const elapsed = performance.now() - start;
          const progress = Math.min(1, elapsed / RECORD_MS);
          setRecording({ phase: 'rec', progress });
          if (progress < 1) {
            requestAnimationFrame(raf);
          } else {
            setRecording(null);
            setPicker({ mode: 'record' });
          }
        };
        requestAnimationFrame(raf);
      }
    };
    tick(3);
  };

  const handlePickClass = (clase) => {
    send({ tipo: 'save_sample', mano, clase });
    setPicker(null);
  };

  const handleRetrain = () => {
    setRetrain({ mano, running: true, epoca: 0, total: 0, loss: 0, acc: 0 });
    send({ tipo: 'retrain', mano, epochs: 40 });
  };

  const canTrain = canTrainMano;

  return (
    <div className="flex flex-col h-full w-full text-white bg-gd-bg">
      {/* ============== HEADER ============== */}
      <header className="h-16 bg-gd-panel border-b border-gd-soft px-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg px-3 py-1.5 text-sm"
            onClick={() => { window.location.hash = '/editor'; }}
          >
            ← Volver
          </button>
          <h1 className="text-lg font-semibold">Entrenamiento</h1>
        </div>

        {/* Selector de mano — botones grandes */}
        <div className="flex gap-2">
          <HandTab
            active={mano === 'dominant'}
            label="Dominante"
            sub="Objetos"
            icon="✊"
            source={sourceMode.dominant}
            onClick={() => setMano('dominant')}
          />
          <HandTab
            active={mano === 'support'}
            label="Apoyo"
            sub="Slides / pausa"
            icon="🫲"
            source={sourceMode.support}
            onClick={() => setMano('support')}
          />
        </div>

        <div className="flex items-center gap-3 text-sm">
          <StatusDot label="Backend" ok={connected} />
          <StatusDot label={mano === 'support' ? 'Apoyo' : 'Dominante'} ok={manoVisible} />
        </div>
      </header>

      {/* ============== CUERPO ============== */}
      <main className="flex-1 flex min-h-0">
        {/* Vista cámara / landmarks (ambas manos, destacando la activa) */}
        <section className="flex-1 p-6 grid place-items-center min-w-0">
          <CameraLandmarks
            landmarksDominant={landmarksDom}
            landmarksSupport={landmarksSup}
            manoActiva={mano}
            manoVisible={manoVisible}
            recording={recording}
          />
        </section>

        {/* Panel derecho: predicción + botones */}
        <aside className="w-96 bg-gd-panel border-l border-gd-soft p-6 flex flex-col gap-5">
          <ValidationPanel
            mano={mano}
            gesto={gestoActual}
            confianza={confianza}
            fuente={fuenteActual}
            manoVisible={manoVisible}
            onCorrect={handleCorrect}
            onWrong={handleWrong}
            onRecord={handleRecord}
          />
        </aside>
      </main>

      {/* ============== DATASET BAR ============== */}
      <footer className="border-t border-gd-soft bg-gd-panel/80 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs uppercase tracking-widest text-white/50">
            Muestras · mano {mano === 'support' ? 'de apoyo' : 'dominante'}
          </div>
          <div className="text-[11px] text-white/50">
            Cambia de mano con los botones de arriba para ver/editar el otro dataset.
          </div>
        </div>
        <DatasetBar stats={stats} />
        <div className="flex justify-between items-center mt-3">
          <div className="text-xs text-white/60">
            Total: {Object.values(stats).reduce((a, b) => a + b, 0)} muestras ·
            mínimo 2 clases con 5+ muestras para reentrenar esta mano.
          </div>
          <button
            className={`px-5 py-2.5 rounded-lg text-sm font-medium ${
              canTrain
                ? 'bg-emerald-600 hover:bg-emerald-500'
                : 'bg-gd-soft/60 text-white/40 cursor-not-allowed'
            }`}
            onClick={canTrain ? handleRetrain : undefined}
            disabled={!canTrain}
            title={canTrain ? `Reentrenar modelo de la mano ${mano === 'support' ? 'de apoyo' : 'dominante'}` : 'Faltan muestras por clase'}
          >
            🔁 Reentrenar {mano === 'support' ? 'apoyo' : 'dominante'}
          </button>
        </div>
      </footer>

      {/* ============== MODALES ============== */}
      {picker && (
        <ClassPicker
          title={picker.mode === 'record' ? '¿Qué gesto grabaste?' : '¿Cuál era el gesto correcto?'}
          clases={clases}
          onPick={handlePickClass}
          onCancel={() => setPicker(null)}
        />
      )}

      {retrain && (
        <RetrainModal
          state={retrain}
          onClose={() => setRetrain(null)}
        />
      )}

      {toast && (
        <div className={`fixed bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg text-sm shadow-xl ${
          toast.kind === 'err' ? 'bg-red-600' : 'bg-emerald-600'
        }`}>
          {toast.text}
        </div>
      )}
    </div>
  );
}

// ===================================================================
// CameraLandmarks — canvas con AMBAS manos. La mano activa sale en
// color pleno; la otra, en gris tenue para no distraer pero dar
// contexto de que sigue siendo visible.
// ===================================================================
function CameraLandmarks({ landmarksDominant, landmarksSupport, manoActiva, manoVisible, recording }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Fondo con rejilla suave.
    ctx.fillStyle = '#0b0f1a';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      ctx.beginPath(); ctx.moveTo((W * i) / 10, 0); ctx.lineTo((W * i) / 10, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (H * i) / 10); ctx.lineTo(W, (H * i) / 10); ctx.stroke();
    }

    const hayAlguna = (landmarksDominant && landmarksDominant.length >= 21)
                  || (landmarksSupport  && landmarksSupport.length  >= 21);

    if (!hayAlguna) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Muestra tu mano frente a la cámara', W / 2, H / 2);
      return;
    }

    // Dibuja primero la mano INACTIVA (atrás, gris), luego la activa (delante, color).
    const manoInactiva = manoActiva === 'support' ? 'dominant' : 'support';
    const parEntries = [
      { lm: manoInactiva === 'dominant' ? landmarksDominant : landmarksSupport, active: false },
      { lm: manoActiva   === 'dominant' ? landmarksDominant : landmarksSupport, active: true  },
    ];

    for (const { lm, active } of parEntries) {
      if (!lm || lm.length < 21) continue;
      const boneColor = active ? '#39a0ff' : 'rgba(120,140,170,0.35)';
      const dotColor  = active ? '#fff'    : 'rgba(200,210,230,0.45)';
      const wristColor = active ? '#ffaa33' : 'rgba(255,170,51,0.45)';
      const boneW = active ? 3 : 2;
      const dotR  = active ? 4 : 3;
      const wristR = active ? 6 : 4;

      ctx.strokeStyle = boneColor;
      ctx.lineWidth = boneW;
      for (const [a, b] of HAND_BONES) {
        ctx.beginPath();
        ctx.moveTo(lm[a][0] * W, lm[a][1] * H);
        ctx.lineTo(lm[b][0] * W, lm[b][1] * H);
        ctx.stroke();
      }
      for (let i = 0; i < lm.length; i++) {
        const [x, y] = lm[i];
        ctx.beginPath();
        ctx.fillStyle = i === 0 ? wristColor : dotColor;
        ctx.arc(x * W, y * H, i === 0 ? wristR : dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [landmarksDominant, landmarksSupport, manoActiva, manoVisible]);

  return (
    <div className="relative w-full max-w-[720px] aspect-video rounded-xl border border-gd-soft overflow-hidden shadow-2xl">
      <canvas ref={canvasRef} width={1280} height={720} className="w-full h-full bg-black" />
      {recording && (
        <div className="absolute inset-0 grid place-items-center bg-black/50 backdrop-blur-sm">
          {recording.phase === 'count' ? (
            <div className="text-center">
              <div className="text-[120px] font-bold leading-none">{recording.seconds}</div>
              <div className="text-sm text-white/70 mt-2">Preparándose…</div>
            </div>
          ) : (
            <div className="text-center w-4/5 max-w-md">
              <div className="text-4xl font-bold text-red-400 mb-5 flex items-center justify-center gap-3">
                <span className="w-4 h-4 rounded-full bg-red-500 animate-pulse" />
                GRABANDO
              </div>
              <div className="h-4 w-full bg-white/10 rounded-full overflow-hidden border border-white/20">
                <div
                  className="h-full bg-gradient-to-r from-red-500 to-red-400"
                  style={{ width: `${(recording.progress || 0) * 100}%` }}
                />
              </div>
              <div className="text-xs text-white/70 mt-3">
                Ejecuta el gesto — incluye el movimiento completo si es temporal.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ===================================================================
// ValidationPanel — muestra la predicción de la mano activa + botones.
// ===================================================================
function ValidationPanel({ mano, gesto, confianza, fuente, manoVisible, onCorrect, onWrong, onRecord }) {
  const icon = gesto ? (CLASS_ICONS[gesto] || '❔') : '—';
  const label = gesto ? (CLASS_LABELS[gesto] || gesto) : 'Sin detección';
  const pct = confianza != null ? Math.round(confianza * 100) : null;
  const manoLabel = mano === 'support' ? 'mano de apoyo' : 'mano dominante';

  return (
    <>
      <div>
        <div className="text-xs uppercase tracking-widest text-white/50 mb-2">
          Gesto detectado · {manoLabel}
        </div>
        <div className="bg-gd-bg rounded-xl p-5 border border-gd-soft">
          <div className="flex items-center gap-4">
            <div className="text-6xl">{icon}</div>
            <div className="flex-1 min-w-0">
              <div className="text-2xl font-semibold truncate">{label}</div>
              <div className="text-xs text-white/60 mt-1">
                {pct != null ? `${pct}%` : '—'}
                {fuente && <span className="ml-2 px-1.5 py-0.5 bg-gd-soft rounded text-[10px] uppercase">{fuente}</span>}
              </div>
            </div>
          </div>
          {!manoVisible && (
            <div className="mt-3 text-xs text-amber-400">
              ⚠ No se detecta tu mano — acércala a la cámara.
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <button
          className="bg-emerald-600 hover:bg-emerald-500 rounded-xl py-4 font-semibold flex flex-col items-center gap-1"
          onClick={onCorrect}
          disabled={!manoVisible}
        >
          <span className="text-2xl">✓</span>
          <span className="text-sm">Correcto</span>
        </button>
        <button
          className="bg-red-600/90 hover:bg-red-500 rounded-xl py-4 font-semibold flex flex-col items-center gap-1"
          onClick={onWrong}
          disabled={!manoVisible}
        >
          <span className="text-2xl">✗</span>
          <span className="text-sm">No es ese</span>
        </button>
      </div>

      <button
        className="bg-gd-accent hover:bg-sky-500 rounded-xl py-4 font-semibold flex items-center justify-center gap-2"
        onClick={onRecord}
        disabled={!manoVisible}
      >
        <span className="text-2xl">●</span>
        <span>Grabar nueva muestra</span>
      </button>

      <div className="text-[11px] text-white/50 leading-relaxed mt-auto">
        <b>Correcto</b> guarda la secuencia actual como el gesto mostrado.<br/>
        <b>No es ese</b> abre un selector para corregir la etiqueta.<br/>
        <b>Grabar</b> captura un gesto nuevo desde cero tras una cuenta regresiva.
      </div>
    </>
  );
}

// ===================================================================
// ClassPicker — grid de tarjetas con las clases entrenables.
// ===================================================================
function ClassPicker({ title, clases, onPick, onCancel }) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 grid place-items-center p-6">
      <div className="bg-gd-panel rounded-2xl border border-gd-soft shadow-2xl max-w-3xl w-full p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-xl font-semibold">{title}</h2>
          <button
            className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg px-3 py-1.5 text-sm"
            onClick={onCancel}
          >Cancelar</button>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {clases.map((c) => (
            <button
              key={c}
              className="bg-gd-bg border border-gd-soft hover:border-gd-accent hover:bg-gd-soft/40 rounded-xl p-4 flex flex-col items-center gap-2 transition"
              onClick={() => onPick(c)}
            >
              <span className="text-4xl">{CLASS_ICONS[c] || '❔'}</span>
              <span className="text-sm">{CLASS_LABELS[c] || c}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ===================================================================
// DatasetBar — barras con conteo por clase.
// ===================================================================
function DatasetBar({ stats }) {
  const entries = Object.entries(stats);
  if (!entries.length) {
    return <div className="text-xs text-white/50">Sin muestras aún. Valida o graba gestos para empezar.</div>;
  }
  const max = Math.max(10, ...entries.map(([, n]) => n));
  return (
    <div className="grid grid-cols-3 gap-x-6 gap-y-2">
      {entries.map(([c, n]) => (
        <div key={c} className="flex items-center gap-2 text-xs">
          <span className="w-4 text-base">{CLASS_ICONS[c] || '•'}</span>
          <span className="flex-1 truncate text-white/70">{CLASS_LABELS[c] || c}</span>
          <div className="w-24 h-2 bg-gd-soft/60 rounded overflow-hidden">
            <div
              className={`h-full ${n >= 5 ? 'bg-emerald-500' : 'bg-amber-500'}`}
              style={{ width: `${Math.min(100, (n / max) * 100)}%` }}
            />
          </div>
          <span className="w-8 text-right text-white/80 font-mono">{n}</span>
        </div>
      ))}
    </div>
  );
}

// ===================================================================
// RetrainModal — progreso del entrenamiento en vivo.
// ===================================================================
function RetrainModal({ state, onClose }) {
  // La barra refleja SIEMPRE las épocas reales ejecutadas / total. Si el
  // early stopping cortó en 25/40, la barra queda al 62%. Es honesto: el
  // badge verde explica por qué se detuvo antes y cuál fue el mejor punto.
  const pct = state.total > 0 ? Math.round((state.epoca / state.total) * 100) : 0;
  const finished = state.done || !!state.error;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 grid place-items-center p-6">
      <div className="bg-gd-panel rounded-2xl border border-gd-soft shadow-2xl max-w-lg w-full p-6">
        <h2 className="text-xl font-semibold mb-4">
          {state.error ? '⚠ Error' : state.done ? '✓ Entrenamiento completado' : 'Entrenando modelo…'}
        </h2>

        {state.error ? (
          <div className="text-sm bg-red-600/30 border border-red-500/60 rounded-lg p-3">
            {state.error}
          </div>
        ) : (
          <>
            <div className="w-full h-4 bg-gd-bg rounded-full overflow-hidden border border-gd-soft">
              <div
                className="h-full bg-gd-accent transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-white/70 mt-1">
              <span>
                {state.done && state.stoppedEarly
                  ? `Mejor modelo en época ${state.bestEpoch} / ${state.total}`
                  : `Época ${state.epoca} / ${state.total || '?'}`}
              </span>
              <span>{pct}%</span>
            </div>
            {state.done && state.stoppedEarly && (
              <div className="mt-3 text-xs bg-emerald-600/20 border border-emerald-500/50 text-emerald-200 rounded-lg px-3 py-2">
                ✓ Early stopping activado: el modelo alcanzó su óptimo en la
                época {state.bestEpoch} y se detuvo antes para no sobre-entrenar.
                Los pesos guardados son los de esa mejor época.
              </div>
            )}
            <div className="grid grid-cols-2 gap-3 mt-4 text-sm">
              {/* Al terminar mostramos las métricas del mejor modelo
                  (son los pesos que quedaron guardados). Durante el
                  entrenamiento mostramos las métricas en vivo. */}
              <Metric label="Loss" value={state.loss} />
              <Metric label="Accuracy" value={state.acc} isPct />
              {state.val_loss != null && (
                <Metric
                  label={state.done ? 'Val loss (mejor)' : 'Val loss'}
                  value={state.done ? state.bestValLoss : state.val_loss}
                />
              )}
              {state.val_acc != null && (
                <Metric
                  label={state.done ? 'Val acc (mejor)' : 'Val acc'}
                  value={state.done ? state.bestValAcc : state.val_acc}
                  isPct
                />
              )}
            </div>
          </>
        )}

        {finished && (
          <button
            className="w-full mt-5 bg-emerald-600 hover:bg-emerald-500 rounded-lg py-2.5 font-medium"
            onClick={onClose}
          >
            Cerrar
          </button>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, isPct }) {
  const v = value == null ? '—' : isPct ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
  return (
    <div className="bg-gd-bg rounded-lg border border-gd-soft p-2">
      <div className="text-[11px] uppercase text-white/50">{label}</div>
      <div className="font-mono">{v}</div>
    </div>
  );
}

function StatusDot({ label, ok }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-500'}`} />
      <span className="text-xs text-white/70">{label}</span>
    </div>
  );
}

// Pestaña grande para elegir mano. Muestra icono, subtítulo y badge con
// la fuente actual del modelo (LSTM / rule-based / sin modelo).
function HandTab({ active, label, sub, icon, source, onClick }) {
  const baseCls = 'flex items-center gap-3 rounded-xl px-4 py-2 transition border';
  const activeCls = 'bg-gd-accent/20 border-gd-accent shadow-[0_0_0_2px_rgba(57,160,255,0.25)]';
  const idleCls = 'bg-gd-bg border-gd-soft hover:bg-gd-soft/40';
  return (
    <button
      className={`${baseCls} ${active ? activeCls : idleCls}`}
      onClick={onClick}
      title={`Cambiar a ${label.toLowerCase()}`}
    >
      <span className="text-2xl">{icon}</span>
      <span className="flex flex-col items-start leading-tight">
        <span className="text-sm font-semibold">{label}</span>
        <span className="text-[10px] text-white/50">{sub}</span>
      </span>
      {source && (
        <span className={`ml-1 px-1.5 py-0.5 rounded text-[9px] uppercase ${
          source === 'lstm' ? 'bg-emerald-600' : 'bg-amber-600'
        }`}>{source}</span>
      )}
    </button>
  );
}
