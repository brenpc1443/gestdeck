// Control.jsx — Ventana privada del expositor (always-on-top).
// - Preview del slide actual con los objetos + posición de la mano.
// - Indicadores de backend, cámara, modo de clasificación.
// - Navegación manual + reset.

import React, { useEffect, useState } from 'react';
import useWebSocket from '../hooks/useWebSocket.js';
import useObjects from '../hooks/useObjects.js';
import useSlideImages from '../hooks/useSlideImages.js';
import SlideCanvas from '../components/SlideCanvas.jsx';
import GestureBadge from '../components/GestureBadge.jsx';

export default function Control() {
  const { connected, lastFrame, lastEvent, send } = useWebSocket();
  const objects = useObjects(lastFrame);
  const { slideImages } = useSlideImages();

  const slideIndex = lastFrame?.slide_actual || 1;
  const currentObjects = objects.filter((o) => o.slide_actual === slideIndex);
  const slideImage = slideImages[slideIndex - 1] || null;
  const totalSlides = lastFrame?.total_slides || slideImages.length || 1;

  // Avisos persistentes que llegan como eventos (no frames)
  const [cameraError, setCameraError] = useState(null);
  const [sourceMode, setSourceMode] = useState(null);   // "lstm" | "rule-based"

  useEffect(() => {
    if (!lastEvent) return;
    if (lastEvent.tipo === 'error_camara') setCameraError(lastEvent.detalle || 'cámara no disponible');
    if (lastEvent.tipo === 'camara_ok')    setCameraError(null);
    if (lastEvent.tipo === 'modo_gesto')   setSourceMode(lastEvent.fuente || null);
  }, [lastEvent]);

  return (
    <div className="p-3 h-screen flex flex-col gap-3 bg-gd-bg text-white text-sm">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Control</h2>
        <span
          className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-red-500'}`}
          title={connected ? 'Backend conectado' : 'Sin backend'}
        />
      </div>

      {cameraError && (
        <div className="bg-red-600/80 text-white text-xs rounded-md px-3 py-2">
          ⚠ Cámara: {cameraError}
        </div>
      )}
      {sourceMode === 'rule-based' && (
        <div className="bg-amber-600/80 text-white text-xs rounded-md px-3 py-2">
          Usando clasificador por reglas (entrena tu LSTM para mejor precisión).
        </div>
      )}

      <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-gd-soft">
        <SlideCanvas
          slideImage={slideImage}
          objects={currentObjects}
          hand={lastFrame?.mano}
          showHand
        />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <GestureBadge
          gesto={lastFrame?.gesto}
          confianza={lastFrame?.confianza}
          fuente={lastFrame?.fuente}
        />
        <div className="text-xs text-white/70">
          FPS: {lastFrame?.camera_fps ?? '—'} &nbsp;|&nbsp;
          Mano: {lastFrame?.mano_detectada ? 'sí' : 'no'}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <button
          className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg py-2 disabled:opacity-40"
          onClick={() => send({ tipo: 'cambiar_slide', slide: Math.max(1, slideIndex - 1) })}
          disabled={slideIndex <= 1}
        >
          ← Anterior
        </button>
        <button
          className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg py-2"
          onClick={() => send({ tipo: 'reiniciar' })}
        >
          Reset (R)
        </button>
        <button
          className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg py-2 disabled:opacity-40"
          onClick={() => send({ tipo: 'cambiar_slide', slide: Math.min(totalSlides, slideIndex + 1) })}
          disabled={slideIndex >= totalSlides}
        >
          Siguiente →
        </button>
      </div>

      <div className="text-xs text-white/60 mt-auto space-y-1">
        <div>Objeto activo: {lastFrame?.objeto_activo || '—'}</div>
        <div>Estado: {lastFrame?.estado_objeto || '—'}</div>
        <div>Slide: {slideIndex} / {totalSlides}</div>
      </div>
    </div>
  );
}
