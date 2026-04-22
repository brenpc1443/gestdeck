// Presenter.jsx — Ventana fullscreen del expositor (Modo Fantasma).
// Solo se ven: el slide de fondo + los objetos interactivos.
// La imagen de slide viene del estado compartido (set por el Editor vía IPC).

import React from 'react';
import useWebSocket from '../hooks/useWebSocket.js';
import useObjects from '../hooks/useObjects.js';
import useSlideImages from '../hooks/useSlideImages.js';
import SlideCanvas from '../components/SlideCanvas.jsx';

export default function Presenter() {
  const { lastFrame } = useWebSocket();
  const objects = useObjects(lastFrame);
  const { slideImages } = useSlideImages();

  const slideIndex = lastFrame?.slide_actual || 1;
  const currentObjects = objects.filter((o) => o.slide_actual === slideIndex);
  const slideImage = slideImages[slideIndex - 1] || null;

  return (
    <div className="w-screen h-screen bg-black">
      <SlideCanvas
        slideImage={slideImage}
        objects={currentObjects}
        hand={null}
        showHand={false}
      />
    </div>
  );
}
