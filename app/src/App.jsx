// App.jsx — router mínimo sin dependencias externas.
// Usa location.hash para distinguir las 3 vistas (editor / presenter / control).

import React, { useEffect, useState } from 'react';
import Editor from './windows/Editor.jsx';
import Presenter from './windows/Presenter.jsx';
import Control from './windows/Control.jsx';
import Training from './windows/Training.jsx';

function getRoute() {
  const h = window.location.hash.replace(/^#/, '') || '/editor';
  if (h.startsWith('/presenter')) return 'presenter';
  if (h.startsWith('/control')) return 'control';
  if (h.startsWith('/training')) return 'training';
  return 'editor';
}

export default function App() {
  const [route, setRoute] = useState(getRoute());

  useEffect(() => {
    const onHash = () => setRoute(getRoute());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  if (route === 'presenter') return <Presenter />;
  if (route === 'control')   return <Control />;
  if (route === 'training')  return <Training />;
  return <Editor />;
}
