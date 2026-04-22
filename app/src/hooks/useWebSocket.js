// useWebSocket.js — hook de conexión al backend Python.
// Auto-reconexión exponencial. Nunca espera respuesta (regla de resiliencia).

import { useCallback, useEffect, useRef, useState } from 'react';

export default function useWebSocket(url = null) {
  const wsUrl = url || (window.gestdeck?.wsUrl ?? 'ws://127.0.0.1:8765');
  const [connected, setConnected] = useState(false);
  const [lastFrame, setLastFrame] = useState(null);
  const [lastEvent, setLastEvent] = useState(null);
  const socketRef = useRef(null);
  const retryRef = useRef(300);

  const connect = useCallback(() => {
    let ws;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      setTimeout(connect, retryRef.current);
      retryRef.current = Math.min(5000, retryRef.current * 1.7);
      return;
    }
    socketRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retryRef.current = 300;
    };
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, retryRef.current);
      retryRef.current = Math.min(5000, retryRef.current * 1.7);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.tipo === 'frame') setLastFrame(msg);
        else setLastEvent(msg);
      } catch {}
    };
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => { if (socketRef.current) socketRef.current.close(); };
  }, [connect]);

  const send = useCallback((msg) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }, []);

  return { connected, lastFrame, lastEvent, send };
}
