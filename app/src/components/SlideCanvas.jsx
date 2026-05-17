// SlideCanvas.jsx — Render del slide + objetos.
// Usa DOM/CSS transforms (más simple de depurar que Three.js).
// La arquitectura permite sustituirlo por WebGL sin cambiar el contrato.

import React from 'react';

// Halos por estado del objeto (Modo Virtual)
const STATE_HALO = {
  EN_MANO:   'drop-shadow(0 0 16px rgba(57,160,255,0.95))',
  FLOTANDO:  'drop-shadow(0 0 18px rgba(170,120,255,0.85))',
  EN_SLIDE:  'none',
};

export default function SlideCanvas({
  slideImage,
  objects = [],
  hand = null,
  showHand = false,
  selectedId = null,
  onObjectMouseDown = null,
  onObjectContextMenu = null,
}) {
  return (
    <div className="relative w-full h-full bg-black overflow-hidden select-none">
      {slideImage && (
        <img
          src={slideImage}
          alt="slide"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          draggable={false}
        />
      )}

      {objects
        .slice()
        .sort((a, b) => (a.z_index || 0) - (b.z_index || 0))
        .map((o) => (
          <ObjectView
            key={o.id}
            o={o}
            selected={o.id === selectedId}
            onMouseDown={onObjectMouseDown ? (e) => onObjectMouseDown(o, e) : undefined}
            onContextMenu={onObjectContextMenu ? (e) => onObjectContextMenu(o, e) : undefined}
          />
        ))}

      {showHand && hand && (
        <div
          className="absolute w-5 h-5 rounded-full border-2 border-cyan-300 pointer-events-none shadow-[0_0_14px_rgba(57,160,255,0.9)]"
          style={{
            left: `${hand.x * 100}%`,
            top: `${hand.y * 100}%`,
            transform: 'translate(-50%, -50%)',
          }}
        />
      )}
    </div>
  );
}

function ObjectView({ o, selected, onMouseDown, onContextMenu }) {
  const [x, y] = o.posicion_actual;
  const src = resolveSrc(o);
  const halo = STATE_HALO[o.estado] || 'none';
  const canInteract = !!(onMouseDown || onContextMenu);

  const cursorCls = canInteract ? 'cursor-grab active:cursor-grabbing' : 'pointer-events-none';
  const ringCls = selected
    ? 'ring-2 ring-offset-2 ring-offset-transparent ring-gd-accent rounded-lg'
    : '';

  return (
    <div
      className={`absolute ${cursorCls} ${ringCls}`}
      style={{
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        transform: `translate(-50%, -50%) scale(${o.escala || 1}) rotate(${(o.rotacion || 0)}rad)`,
        transition: 'filter 200ms ease, box-shadow 200ms ease',
        filter: halo,
      }}
      onMouseDown={onMouseDown}
      onContextMenu={onContextMenu}
    >
      {o.tipo === 'text' ? (
        <div className="px-4 py-2 bg-gd-accent/80 rounded-lg text-white font-semibold whitespace-nowrap">
          {o.meta?.text || o.id}
        </div>
      ) : src ? (
        <img src={src} alt={o.id} className="w-28 h-28 object-contain pointer-events-none" draggable={false} />
      ) : (
        <div className="w-20 h-20 rounded-2xl bg-gd-accent/70 text-white grid place-items-center text-xs">
          {o.id}
        </div>
      )}
    </div>
  );
}

function resolveSrc(o) {
  if (!o.archivo) return null;
  // URL absoluta ya resuelta
  if (/^(https?:|data:|blob:|file:|gestdeck:)/.test(o.archivo)) return o.archivo;
  // Ruta relativa al proyecto → gestdeck://
  return `gestdeck://${o.archivo.replace(/^\/+/, '')}`;
}
