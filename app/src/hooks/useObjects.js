// useObjects.js — estado interpolado de los objetos del slide.
// El backend envía posiciones ~30fps; aquí interpolamos a 60fps para suavidad.
//
// El factor de lerp es ADAPTATIVO según el estado del objeto:
//   - EN_MANO / FLOTANDO (controlado por la mano en tiempo real) → lerp casi 1.0
//     para que el objeto siga la mano sin retraso perceptible.
//   - EN_SLIDE / CONGELADO (en reposo o con inercia física) → lerp moderado
//     que suaviza la animación de la inercia sin añadir lag.
// Antes era 0.35 fijo para todo, lo que añadía ~120 ms de lag al arrastre.

import { useEffect, useRef, useState } from 'react';

const LERP_DRAG = 0.9;       // EN_MANO / FLOTANDO: snap a la mano
const LERP_PHYSICS = 0.6;    // EN_SLIDE / CONGELADO: suaviza inercia
const EPSILON = 5e-4;        // si la diferencia es menor, copia el target tal cual

export default function useObjects(latestFrame) {
  const [objects, setObjects] = useState([]);
  const targetRef = useRef(new Map());

  useEffect(() => {
    if (!latestFrame || !Array.isArray(latestFrame.objetos)) return;
    const map = new Map();
    for (const o of latestFrame.objetos) {
      map.set(o.id, o);
    }
    targetRef.current = map;
  }, [latestFrame]);

  useEffect(() => {
    let raf;
    const step = () => {
      const targets = targetRef.current;
      setObjects((prev) => {
        const byId = new Map(prev.map((o) => [o.id, o]));
        const next = [];
        for (const [id, tgt] of targets) {
          const cur = byId.get(id);
          if (!cur) {
            next.push({ ...tgt });
            continue;
          }
          const lerp = (tgt.estado === 'EN_MANO' || tgt.estado === 'FLOTANDO')
            ? LERP_DRAG
            : LERP_PHYSICS;
          const [tx, ty] = tgt.posicion_actual;
          const [cx, cy] = cur.posicion_actual;
          const dx = tx - cx;
          const dy = ty - cy;
          const dscale = (tgt.escala || 1) - (cur.escala || 1);
          // Si ya estamos prácticamente en el target, snap directo y evita
          // el coste de re-renderizar por diferencias subpíxel.
          if (Math.abs(dx) < EPSILON && Math.abs(dy) < EPSILON && Math.abs(dscale) < EPSILON) {
            next.push(tgt);
          } else {
            next.push({
              ...tgt,
              posicion_actual: [cx + dx * lerp, cy + dy * lerp],
              escala: (cur.escala || 1) + dscale * lerp,
            });
          }
        }
        return next;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  return objects;
}
