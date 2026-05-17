// GestureBadge.jsx — indicador visual del gesto activo + confianza.
import React from 'react';

const COLORS = {
  PUNO_CERRADO: 'bg-red-500/80',
  PALMA_ABIERTA: 'bg-emerald-500/80',
  PALMA_VERTICAL: 'bg-teal-500/80',
  INDICE: 'bg-sky-500/80',
  PELLIZCO: 'bg-amber-500/80',
  PELLIZCO_INV: 'bg-orange-500/80',
  EMPUJE: 'bg-fuchsia-500/80',
  SIGUIENTE: 'bg-indigo-500/80',
  ANTERIOR: 'bg-indigo-500/80',
  PAUSA: 'bg-rose-500/80',
  NINGUNO: 'bg-gray-600/70',
};

export default function GestureBadge({ gesto, confianza, fuente }) {
  const color = COLORS[gesto] || 'bg-gray-600/70';
  return (
    <div className={`px-3 py-1.5 rounded-full text-xs font-medium text-white ${color} inline-flex items-center gap-2`}>
      <span>{gesto || '—'}</span>
      {typeof confianza === 'number' && (
        <span className="opacity-80">{Math.round(confianza * 100)}%</span>
      )}
      {fuente && (
        <span className="opacity-60 border-l border-white/30 pl-2 ml-1">{fuente}</span>
      )}
    </div>
  );
}
