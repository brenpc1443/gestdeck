// useSlideImages.js — suscripción al estado de slides compartido por el main
// process de Electron. Las tres ventanas (Editor/Presenter/Control) lo usan
// para pintar la imagen del slide actual como fondo del SlideCanvas.

import { useEffect, useState } from 'react';

export default function useSlideImages() {
  const [state, setState] = useState({ slideImages: [], sessionName: null, slidesDir: null });

  useEffect(() => {
    if (!window.gestdeck?.getActiveSlides) return;
    let active = true;
    window.gestdeck.getActiveSlides().then((s) => {
      if (active && s) setState(s);
    });
    const off = window.gestdeck.onSlidesUpdated?.((s) => {
      if (active && s) setState(s);
    });
    return () => {
      active = false;
      if (typeof off === 'function') off();
    };
  }, []);

  return state;
}
