// useObjects.js — estado interpolado de los objetos del slide.
// El backend envía posiciones ~30fps; aquí interpolamos a 60fps para suavidad.

import { useEffect, useRef, useState } from 'react';

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
          } else {
            // interpolación suave (lerp) hacia el target
            const lerp = 0.35;
            const [tx, ty] = tgt.posicion_actual;
            const [cx, cy] = cur.posicion_actual;
            next.push({
              ...tgt,
              posicion_actual: [
                cx + (tx - cx) * lerp,
                cy + (ty - cy) * lerp,
              ],
              escala: cur.escala + (tgt.escala - cur.escala) * lerp,
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
