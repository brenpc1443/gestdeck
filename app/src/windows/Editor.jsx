// Editor.jsx — Ventana principal de configuración de sesión.
// Flujo:
//   1. Cargar PPTX → se convierte a PNGs y se crea una sesión nueva.
//      (o cargar config.json existente).
//   2. Arrastrar objetos desde la biblioteca al slide activo.
//   3. Click sobre objeto = seleccionar. Click+drag = mover. Right-click = eliminar.
//   4. Panel derecho con inspector del objeto seleccionado.
//   5. Guardar sesión (config.json).
//   6. Iniciar presentación → abre ventanas Presenter + Control.
//
// Atajos:
//   ←/→      Navegar entre slides
//   R        Reset objetos
//   Supr     Eliminar objeto seleccionado
//   +/−      Escalar objeto seleccionado
//
// El Editor NUNCA muta el .pptx original. Todo se guarda en sessions/<nombre>/.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useWebSocket from '../hooks/useWebSocket.js';
import useObjects from '../hooks/useObjects.js';
import SlideCanvas from '../components/SlideCanvas.jsx';
import GestureBadge from '../components/GestureBadge.jsx';

const OBJECT_LIBRARY = [
  { tipo: 'svg',  archivo: 'assets/objects/arrow.svg',  label: 'Flecha'   },
  { tipo: 'svg',  archivo: 'assets/objects/circle.svg', label: 'Círculo'  },
  { tipo: 'svg',  archivo: 'assets/objects/square.svg', label: 'Cuadrado' },
  { tipo: 'text', archivo: null,                        label: 'Texto'    },
];

// Genera un id único por objeto añadido
let _oid = 1;
const newId = (prefix = 'obj') => `${prefix}_${Date.now().toString(36)}_${_oid++}`;

export default function Editor() {
  const { connected, lastFrame, lastEvent, send } = useWebSocket();
  const liveObjects = useObjects(lastFrame);

  const [session, setSession] = useState({
    configPath: null,              // ruta absoluta a config.json
    sessionName: null,
    pptxPath: null,
    slidesDir: null,               // ruta relativa gestdeck://...
    slideImages: [],               // array de URLs gestdeck:// (index = slide-1)
    totalSlides: 0,
  });
  const [slideIndex, setSlideIndex] = useState(1);
  const [dominant, setDominant] = useState('derecha');
  const [busy, setBusy] = useState(null);             // string con texto de acción
  const [selectedId, setSelectedId] = useState(null);
  const [toast, setToast] = useState(null);           // { text, kind }
  const [cameraError, setCameraError] = useState(null);
  const [sourceMode, setSourceMode] = useState(null); // "lstm" | "rule-based"

  const canvasRef = useRef(null);
  const dragStateRef = useRef(null);

  // --- Eventos del backend --------------------------------------------
  useEffect(() => {
    if (!lastEvent) return;
    switch (lastEvent.tipo) {
      case 'sesion_cargada':
        setSession((s) => ({ ...s, totalSlides: lastEvent.slides }));
        setDominant(lastEvent.dominante || 'derecha');
        break;
      case 'session_ready':
        setSession((s) => ({ ...s, totalSlides: lastEvent.slides }));
        break;
      case 'sesion_guardada':
        setToast({ text: `💾 Guardado: ${lastEvent.path}`, kind: 'ok' });
        break;
      case 'objeto_eliminado':
        if (selectedId === lastEvent.id) setSelectedId(null);
        break;
      case 'reiniciado_ok':
        setToast({ text: '↻ Objetos reseteados', kind: 'ok' });
        break;
      case 'error_camara':
        setCameraError(lastEvent.detalle || 'cámara no disponible');
        break;
      case 'camara_ok':
        setCameraError(null);
        break;
      case 'modo_gesto':
        setSourceMode(lastEvent.fuente || null);
        break;
      case 'error':
        setToast({ text: `⚠ ${lastEvent.detalle}`, kind: 'err' });
        break;
      default:
        break;
    }
  }, [lastEvent, selectedId]);

  // Auto-ocultar toast
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2800);
    return () => clearTimeout(t);
  }, [toast]);

  // --- Objetos del slide activo ---------------------------------------
  const currentObjects = useMemo(
    () => liveObjects.filter((o) => o.slide_actual === slideIndex),
    [liveObjects, slideIndex]
  );
  const selectedObject = useMemo(
    () => liveObjects.find((o) => o.id === selectedId) || null,
    [liveObjects, selectedId]
  );

  // --- Acciones de sesión ---------------------------------------------
  const handleLoadPptx = useCallback(async () => {
    const pptxPath = await window.gestdeck.openPptxDialog();
    if (!pptxPath) return;

    const base = pptxPath.split(/[\\/]/).pop().replace(/\.pptx?$/i, '');
    const sessionName = base.toLowerCase().replace(/[^a-z0-9_-]+/g, '_') || 'nueva_sesion';

    setBusy('Convirtiendo PPTX a imágenes…');
    try {
      const res = await window.gestdeck.convertPptx(pptxPath, sessionName);
      const slides = res.urls;
      const slidesDir = res.sessionDir + '/slides';
      setSession((s) => ({
        ...s,
        configPath: null,
        sessionName,
        pptxPath,
        slidesDir,
        slideImages: slides,
        totalSlides: slides.length,
      }));
      setSlideIndex(1);
      setSelectedId(null);

      window.gestdeck.setActiveSlides?.({ slideImages: slides, sessionName, slidesDir });

      send({
        tipo: 'setup_session',
        nombre: sessionName,
        total_slides: slides.length,
        pptx_path: pptxPath,
        slides_dir: slidesDir,
      });
      setToast({ text: `✓ ${slides.length} slides importados`, kind: 'ok' });
    } catch (e) {
      alert(`Error convirtiendo PPTX:\n${e.message || e}`);
    } finally {
      setBusy(null);
    }
  }, [send]);

  const handleLoadConfig = useCallback(async () => {
    const configPath = await window.gestdeck.openConfigDialog();
    if (!configPath) return;
    setBusy('Cargando sesión…');
    try {
      const raw = await window.gestdeck.readConfig(configPath);
      const cfg = JSON.parse(raw);
      const slidesDir = cfg.sesion?.slides_dir;
      let slideImages = [];
      if (slidesDir) {
        slideImages = await window.gestdeck.listSlidesInDir(slidesDir);
      }
      const sessionName = cfg.sesion?.nombre || 'sesion';
      setSession({
        configPath,
        sessionName,
        pptxPath: cfg.sesion?.pptx_path || null,
        slidesDir: slidesDir || null,
        slideImages,
        totalSlides: cfg.sesion?.total_slides || (cfg.slides?.length ?? 1),
      });
      setSlideIndex(1);
      setSelectedId(null);
      setDominant(cfg.sesion?.mano_dominante || 'derecha');
      window.gestdeck.setActiveSlides?.({ slideImages, sessionName, slidesDir: slidesDir || null });
      send({ tipo: 'cargar_sesion', path: configPath });
    } catch (e) {
      alert(`Error cargando sesión:\n${e.message || e}`);
    } finally {
      setBusy(null);
    }
  }, [send]);

  const handleSave = useCallback(async () => {
    let configPath = session.configPath;
    if (!configPath) {
      configPath = await window.gestdeck.saveDialog(session.sessionName || 'nueva_sesion');
      if (!configPath) return;
    }
    send({ tipo: 'guardar_sesion', path: configPath });
    setSession((s) => ({ ...s, configPath }));
  }, [session, send]);

  // --- Biblioteca: drag de un item ------------------------------------
  const handleLibraryDragStart = (e, item) => {
    e.dataTransfer.setData('application/x-gestdeck-library', JSON.stringify(item));
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleCanvasDragOver = (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; };

  const handleCanvasDrop = (e) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('application/x-gestdeck-library');
    if (!data) return;
    const item = JSON.parse(data);
    const { x, y } = posFromEvent(e, canvasRef.current);

    // Para texto: pedir al usuario el contenido
    let meta = {};
    if (item.tipo === 'text') {
      const txt = window.prompt('Texto del objeto:', 'Nuevo texto');
      if (!txt) return;
      meta = { text: txt };
    }

    const obj = {
      id: newId(item.tipo),
      tipo: item.tipo,
      archivo: item.archivo,
      posicion_original: [x, y],
      posicion_actual: [x, y],
      escala: 1.0,
      rotacion: 0.0,
      slide_origen: slideIndex,
      slide_actual: slideIndex,
      estado: 'EN_SLIDE',
      z_index: 1,
      meta,
    };
    send({ tipo: 'agregar_objeto', slide: slideIndex, objeto: obj });
    setSelectedId(obj.id);
  };

  // --- Interacción con objetos ----------------------------------------
  const handleObjectMouseDown = (obj, e) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedId(obj.id);
    const rect = canvasRef.current.getBoundingClientRect();
    dragStateRef.current = { id: obj.id, startRect: rect, moved: false };
    const move = (ev) => {
      if (!dragStateRef.current) return;
      dragStateRef.current.moved = true;
      const r = dragStateRef.current.startRect;
      const x = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
      const y = Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height));
      send({ tipo: 'mover_objeto', id: dragStateRef.current.id, x, y, update_origin: true });
    };
    const up = () => {
      dragStateRef.current = null;
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  const handleObjectContextMenu = (obj, e) => {
    e.preventDefault();
    if (confirm(`¿Eliminar objeto "${obj.id}"?`)) {
      send({ tipo: 'eliminar_objeto', id: obj.id });
    }
  };

  const handleCanvasClick = () => setSelectedId(null);

  // --- Navegación -----------------------------------------------------
  const gotoSlide = useCallback((i) => {
    const total = session.totalSlides || 1;
    const clamped = Math.max(1, Math.min(total, i));
    if (clamped === slideIndex) return;
    setSlideIndex(clamped);
    setSelectedId(null);
    send({ tipo: 'cambiar_slide', slide: clamped });
  }, [session.totalSlides, slideIndex, send]);

  // --- Atajos de teclado ----------------------------------------------
  useEffect(() => {
    const onKey = (e) => {
      // Ignora si está tecleando en un input
      if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowRight') { gotoSlide(slideIndex + 1); e.preventDefault(); }
      else if (e.key === 'ArrowLeft')  { gotoSlide(slideIndex - 1); e.preventDefault(); }
      else if (e.key === 'r' || e.key === 'R') { send({ tipo: 'reiniciar' }); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        send({ tipo: 'eliminar_objeto', id: selectedId });
        e.preventDefault();
      } else if ((e.key === '+' || e.key === '=') && selectedId && selectedObject) {
        send({ tipo: 'escalar_objeto', id: selectedId, escala: selectedObject.escala * 1.1 });
      } else if (e.key === '-' && selectedId && selectedObject) {
        send({ tipo: 'escalar_objeto', id: selectedId, escala: selectedObject.escala * 0.9 });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gotoSlide, slideIndex, selectedId, selectedObject, send]);

  const slideImage = session.slideImages[slideIndex - 1] || null;
  const totalSlides = session.totalSlides || 0;

  return (
    <div className="flex h-full w-full text-white">
      {/* ==================== SIDEBAR IZQUIERDA ==================== */}
      <aside className="w-72 bg-gd-panel border-r border-gd-soft p-4 flex flex-col gap-4 overflow-auto">
        <div>
          <h2 className="text-lg font-semibold">GestDeck</h2>
          <p className="text-xs text-white/60">Modo Virtual · Ghost</p>
        </div>

        <div className="grid grid-cols-1 gap-2">
          <button
            className="bg-gd-accent hover:bg-sky-500 rounded-lg py-2 font-medium"
            onClick={handleLoadPptx}
            disabled={!!busy}
          >
            📥 Cargar PPTX
          </button>
          <button
            className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg py-2 text-sm"
            onClick={handleLoadConfig}
            disabled={!!busy}
          >
            Abrir sesión (.json)
          </button>
          <button
            className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg py-2 text-sm"
            onClick={handleSave}
            disabled={!!busy || !session.sessionName}
          >
            💾 Guardar sesión
          </button>
        </div>

        <div className="text-xs text-white/60 border-y border-gd-soft py-2 space-y-0.5">
          {session.sessionName ? (
            <>
              <div><span className="text-white/40">Sesión:</span> {session.sessionName}</div>
              <div><span className="text-white/40">Slides:</span> {totalSlides}</div>
              {session.configPath && (
                <div className="truncate" title={session.configPath}>
                  💾 …/{session.configPath.split(/[\\/]/).slice(-2).join('/')}
                </div>
              )}
            </>
          ) : (
            <div className="text-white/50 italic">Sin sesión cargada</div>
          )}
        </div>

        <StatusRow label="Backend" ok={connected} />
        <StatusRow label="Cámara" ok={!cameraError && !!lastFrame?.camera_fps} />
        <StatusRow label="Iluminación" ok={lastFrame?.iluminacion_ok} />
        <div className="text-xs text-white/60">
          FPS: {lastFrame?.camera_fps ?? '—'}
          {sourceMode && (
            <span className="ml-2 px-1.5 py-0.5 rounded bg-gd-soft text-[10px] uppercase">
              {sourceMode}
            </span>
          )}
        </div>

        {cameraError && (
          <div className="bg-red-600/80 text-xs rounded-md px-2 py-1.5">
            ⚠ Cámara: {cameraError}
          </div>
        )}

        <div>
          <label className="text-xs text-white/60">Mano dominante</label>
          <div className="flex gap-1 mt-1">
            {['derecha', 'izquierda'].map((m) => (
              <button
                key={m}
                className={`flex-1 py-1.5 rounded-md text-sm capitalize ${dominant === m ? 'bg-gd-accent' : 'bg-gd-soft hover:bg-gd-soft/70'}`}
                onClick={() => { setDominant(m); send({ tipo: 'set_mano_dominante', value: m }); }}
              >{m}</button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-xs uppercase tracking-widest text-white/50 mb-2">Objetos</h3>
          <div className="grid grid-cols-2 gap-2">
            {OBJECT_LIBRARY.map((item) => (
              <div
                key={item.label}
                className="bg-gd-soft/60 hover:bg-gd-soft cursor-grab rounded-lg p-3 text-center text-xs select-none"
                draggable
                onDragStart={(e) => handleLibraryDragStart(e, item)}
                title="Arrastra al slide para colocarlo"
              >
                {item.archivo ? (
                  <img
                    src={`gestdeck://${item.archivo}`}
                    alt={item.label}
                    className="w-12 h-12 mx-auto object-contain mb-1 pointer-events-none"
                  />
                ) : (
                  <div className="w-12 h-12 mx-auto grid place-items-center bg-gd-accent/40 rounded mb-1">T</div>
                )}
                <div>{item.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-auto flex flex-col gap-2">
          <button
            className="bg-emerald-600 hover:bg-emerald-500 rounded-lg py-2 font-medium disabled:opacity-40"
            onClick={() => window.gestdeck?.openPresenter?.()}
            disabled={!session.sessionName}
            title={session.sessionName ? 'Abre la ventana Presentador + Control' : 'Carga una sesión primero'}
          >
            ▶ Iniciar presentación
          </button>
          <button
            className="bg-gd-soft/60 hover:bg-gd-soft/40 rounded-lg py-2 text-sm"
            onClick={() => send({ tipo: 'reiniciar' })}
          >
            ↻ Reset objetos
          </button>
        </div>
      </aside>

      {/* ==================== MAIN ==================== */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-14 bg-gd-panel/80 border-b border-gd-soft px-4 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <button
              className="bg-gd-soft hover:bg-gd-soft/70 w-8 h-8 rounded-md disabled:opacity-40"
              onClick={() => gotoSlide(slideIndex - 1)}
              disabled={slideIndex <= 1}
              title="Slide anterior (←)"
            >‹</button>
            <div className="text-sm text-white/70">
              Slide {slideIndex} / {totalSlides || '—'}
            </div>
            <button
              className="bg-gd-soft hover:bg-gd-soft/70 w-8 h-8 rounded-md disabled:opacity-40"
              onClick={() => gotoSlide(slideIndex + 1)}
              disabled={slideIndex >= totalSlides}
              title="Slide siguiente (→)"
            >›</button>
          </div>
          <div className="flex items-center gap-2">
            {lastFrame?.paused && (
              <div className="bg-rose-600/90 text-white text-xs font-semibold rounded-md px-2.5 py-1 flex items-center gap-1.5 animate-pulse">
                <span>⏸</span>
                <span>PAUSADO</span>
              </div>
            )}
            <button
              className="bg-gd-soft hover:bg-gd-soft/70 rounded-lg px-3 py-1.5 text-sm flex items-center gap-1.5"
              onClick={() => { window.location.hash = '/training'; }}
              title="Entrenar / corregir el reconocimiento de gestos"
            >
              🎯 <span>Entrenamiento</span>
            </button>
            <GestureBadge
              gesto={lastFrame?.gesto}
              confianza={lastFrame?.confianza}
              fuente={lastFrame?.fuente}
            />
            <GestureBadge
              gesto={lastFrame?.gesto_apoyo}
              confianza={lastFrame?.confianza_apoyo}
              fuente={lastFrame?.fuente_apoyo}
            />
          </div>
        </header>

        <section className="flex-1 p-6 overflow-auto grid place-items-center">
          <div
            ref={canvasRef}
            className="relative w-full max-w-[1200px] aspect-video bg-gray-900 rounded-lg overflow-hidden border border-gd-soft shadow-2xl"
            onDragOver={handleCanvasDragOver}
            onDrop={handleCanvasDrop}
            onClick={handleCanvasClick}
          >
            <SlideCanvas
              slideImage={slideImage}
              objects={currentObjects}
              hand={lastFrame?.mano}
              showHand
              selectedId={selectedId}
              onObjectMouseDown={handleObjectMouseDown}
              onObjectContextMenu={handleObjectContextMenu}
            />
            {!slideImage && totalSlides === 0 && (
              <div className="absolute inset-0 grid place-items-center pointer-events-none">
                <div className="text-center text-white/40">
                  <div className="text-5xl mb-3">🖼️</div>
                  <div>Carga un PPTX para empezar</div>
                </div>
              </div>
            )}
            {busy && (
              <div className="absolute inset-0 grid place-items-center bg-black/60 backdrop-blur-sm">
                <div className="text-sm">{busy}</div>
              </div>
            )}
          </div>
        </section>

        {/* Tira de miniaturas */}
        {totalSlides > 0 && (
          <div className="h-20 bg-gd-panel/60 border-t border-gd-soft overflow-x-auto flex items-center gap-2 px-3">
            {session.slideImages.map((url, i) => (
              <button
                key={url + i}
                className={`relative h-14 aspect-video rounded overflow-hidden flex-shrink-0 border-2 transition ${
                  i + 1 === slideIndex ? 'border-gd-accent shadow-[0_0_0_2px_rgba(57,160,255,0.4)]' : 'border-gd-soft/40 hover:border-gd-soft'
                }`}
                onClick={() => gotoSlide(i + 1)}
                title={`Slide ${i + 1}`}
              >
                <img src={url} alt={`slide-${i + 1}`} className="w-full h-full object-cover" />
                <span className="absolute bottom-0 right-0 bg-black/70 text-[10px] px-1">{i + 1}</span>
              </button>
            ))}
          </div>
        )}

        <footer className="h-10 bg-gd-panel/60 border-t border-gd-soft px-4 text-xs text-white/60 flex items-center justify-between">
          <span>
            ← → slide · click+drag mover · right-click eliminar · R reset · +/− escalar
          </span>
          <span>Objetos en slide: {currentObjects.length}</span>
        </footer>
      </main>

      {/* ==================== INSPECTOR DERECHA ==================== */}
      {selectedObject && (
        <aside className="w-64 bg-gd-panel border-l border-gd-soft p-4 flex flex-col gap-3 text-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-xs uppercase tracking-widest text-white/50">Objeto</h3>
            <button
              className="text-white/50 hover:text-white text-xs"
              onClick={() => setSelectedId(null)}
              title="Deseleccionar (Esc)"
            >✕</button>
          </div>

          <div className="text-xs space-y-1">
            <div><span className="text-white/40">ID:</span> <span className="font-mono">{selectedObject.id}</span></div>
            <div><span className="text-white/40">Tipo:</span> {selectedObject.tipo}</div>
            <div><span className="text-white/40">Estado:</span> {selectedObject.estado}</div>
            <div><span className="text-white/40">Slide:</span> {selectedObject.slide_actual}</div>
            <div>
              <span className="text-white/40">Pos:</span>{' '}
              ({selectedObject.posicion_actual[0].toFixed(2)},{' '}
              {selectedObject.posicion_actual[1].toFixed(2)})
            </div>
          </div>

          <div>
            <label className="text-xs text-white/60">Escala: {selectedObject.escala.toFixed(2)}×</label>
            <input
              type="range"
              min={0.3}
              max={3}
              step={0.05}
              value={selectedObject.escala}
              onChange={(e) => send({ tipo: 'escalar_objeto', id: selectedObject.id, escala: Number(e.target.value) })}
              className="w-full"
            />
          </div>

          <div className="flex gap-1">
            <button
              className="flex-1 bg-gd-soft hover:bg-gd-soft/70 rounded-md py-1.5 text-xs"
              onClick={() => send({ tipo: 'escalar_objeto', id: selectedObject.id, escala: selectedObject.escala * 0.9 })}
            >− escala</button>
            <button
              className="flex-1 bg-gd-soft hover:bg-gd-soft/70 rounded-md py-1.5 text-xs"
              onClick={() => send({ tipo: 'escalar_objeto', id: selectedObject.id, escala: selectedObject.escala * 1.1 })}
            >+ escala</button>
          </div>

          <button
            className="bg-red-600/80 hover:bg-red-500 rounded-md py-1.5 text-sm"
            onClick={() => {
              if (confirm(`¿Eliminar "${selectedObject.id}"?`)) {
                send({ tipo: 'eliminar_objeto', id: selectedObject.id });
              }
            }}
          >
            🗑 Eliminar (Supr)
          </button>

          <div className="text-[11px] text-white/40 mt-auto">
            Arrastra el objeto con el ratón para moverlo.
            Las teclas +/− escalan el objeto seleccionado.
          </div>
        </aside>
      )}

      {/* ==================== TOAST ==================== */}
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

// ---------------------------------------------------------------
function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-white/70">{label}</span>
      <span className={`w-2.5 h-2.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-500'}`} />
    </div>
  );
}

function posFromEvent(e, container) {
  const r = container.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
    y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
  };
}
