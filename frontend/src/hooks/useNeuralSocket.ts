import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ConnectionStatus, Message, SessionSummary, StoredMessage } from '../types';

const DEFAULT_WS_PORT = 8000;
const getSocketUrl = () => {
  const envUrl = import.meta.env.VITE_WS_URL;
  if (envUrl) return envUrl;
  const protocol = 'wss';
  const host = window.location.hostname || 'localhost';
  return `${protocol}://${host}:${DEFAULT_WS_PORT}/ws`;
};

const SOCKET_URL = getSocketUrl();
const DEFAULT_API_PORT = 8000;
const getApiCandidates = () => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl) return [envUrl];

  const host = window.location.hostname || 'localhost';
  const isSecurePage = window.location.protocol === 'https:';
  const primaryProto = isSecurePage ? 'https' : 'http';
  const primary = `${primaryProto}://${host}:${DEFAULT_API_PORT}`;
  const alt = `${primaryProto === 'https' ? 'http' : 'https'}://${host}:${DEFAULT_API_PORT}`;
  return [primary, alt];
};

export const useNeuralSocket = () => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.CLOSED);
  const [messages, setMessages] = useState<Message[]>([]);
  const [aiIsSpeaking, setAiIsSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  
  const socketRef = useRef<WebSocket | null>(null);
  const apiBaseRef = useRef<string | null>(null);
  const pendingSessionRef = useRef<number | null>(null);
  const currentSessionIdRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlayingRef = useRef(false);
  const apiCandidates = useMemo(() => getApiCandidates(), []);

  const persistSessionId = useCallback((id: number | null) => {
    currentSessionIdRef.current = id;
    setCurrentSessionId(id);
    if (id) localStorage.setItem('neural_session_id', String(id));
    else localStorage.removeItem('neural_session_id');
  }, []);

  const callApi = useCallback(async (path: string, init?: RequestInit) => {
    const bases = apiBaseRef.current
      ? [apiBaseRef.current, ...apiCandidates.filter((b) => b !== apiBaseRef.current)]
      : apiCandidates;

    let lastError: unknown = null;
    for (const base of bases) {
      try {
        const res = await fetch(`${base}${path}`, init);
        if (res.ok) {
          apiBaseRef.current = base;
          return res;
        }
        lastError = new Error(`API ${res.status} ${res.statusText}`);
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError ?? new Error('API call failed');
  }, [apiCandidates]);

  const mapStoredMessage = useCallback((m: StoredMessage): Message => ({
    id: (m.id ?? `${Date.now()}${Math.random()}`).toString(),
    text: m.content,
    sender: m.role === 'assistant' ? 'ai' : 'user',
    timestamp: m.created_at ? m.created_at * 1000 : Date.now()
  }), []);

  const loadSessionWindow = useCallback(async (sessionId: number) => {
    setIsLoadingSession(true);
    try {
      const res = await callApi(`/sessions/${sessionId}/window`);
      const data = await res.json();
      if (data?.anchors && data?.recents) {
        const combined = [...data.anchors, ...data.recents].map(mapStoredMessage);
        setMessages(combined);
      } else {
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to load session window', err);
    } finally {
      setIsLoadingSession(false);
    }
  }, [callApi, mapStoredMessage]);

  const refreshSessions = useCallback(async () => {
    try {
      const res = await callApi('/sessions');
      const data = await res.json();
      if (Array.isArray(data?.sessions)) {
        setSessions(data.sessions as SessionSummary[]);
      }
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    }
  }, [callApi]);

  const switchSession = useCallback(async (sessionId: number) => {
    if (!sessionId) return;
    pendingSessionRef.current = sessionId;
    persistSessionId(sessionId);
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setAiIsSpeaking(false);
    setIsProcessing(false);
    setMessages([]);

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(`SESSION:${sessionId}`);
    }
    await loadSessionWindow(sessionId);
    await refreshSessions();
  }, [loadSessionWindow, persistSessionId, refreshSessions]);

  const createNewSession = useCallback(async (title?: string) => {
    try {
      const res = await callApi('/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      });
      const data = await res.json();
      if (data?.id) {
        await switchSession(data.id);
        return data.id as number;
      }
    } catch (err) {
      console.error('Failed to create session', err);
    }
    return null;
  }, [callApi, switchSession]);

  useEffect(() => {
    const saved = localStorage.getItem('neural_session_id');
    const parsed = saved ? parseInt(saved, 10) : NaN;
    if (!Number.isNaN(parsed)) {
      persistSessionId(parsed);
      loadSessionWindow(parsed);
    }
    refreshSessions();
  }, [loadSessionWindow, persistSessionId, refreshSessions]);

  // --- SOCKET URL LOGIC ---
  const socketCandidates = useMemo(() => {
    const primary = SOCKET_URL;
    if (import.meta.env.VITE_WS_URL) return [primary];
    try {
      const url = new URL(primary);
      const isSecure = primary.startsWith('wss:');
      const alt = `${isSecure ? 'ws' : 'wss'}://${url.host}${url.pathname}`;
      return [primary, alt];
    } catch { return [primary]; }
  }, []);

  // --- AI AUDIO PLAYBACK ---
  const playNextChunk = useCallback(async () => {
    if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    if (audioContextRef.current.state === 'suspended') {
      try {
        await audioContextRef.current.resume();
      } catch (err) {
        console.error('AudioContext resume failed:', err);
      }
    }
    
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    setAiIsSpeaking(true);

    try {
      const chunk = audioQueueRef.current.shift()!;
      const audioBuffer = await audioContextRef.current.decodeAudioData(chunk);
      const source = audioContextRef.current.createBufferSource();
      activeSourceRef.current = source;
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      
      source.onended = () => {
        if (activeSourceRef.current === source) activeSourceRef.current = null;
        isPlayingRef.current = false;
        if (audioQueueRef.current.length === 0) setAiIsSpeaking(false);
        playNextChunk();
      };
      
      source.start(0);
    } catch (err) {
      console.error('Audio decode error:', err);
      activeSourceRef.current = null;
      isPlayingRef.current = false;
      playNextChunk();
    }
  }, []);

  const handleBinaryMessage = useCallback((data: Blob) => {
    setIsProcessing(false); 
    
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        audioQueueRef.current.push(reader.result);
        playNextChunk();
      }
    };
    reader.readAsArrayBuffer(data);
  }, [playNextChunk]);

  // --- USER RECORDING ---
  const startRecording = useCallback(async () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
    
    // Reset states
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setAiIsSpeaking(false);
    setIsProcessing(false); // Reset processing if we start talking again

    if (audioContextRef.current?.state === 'suspended') {
        audioContextRef.current.resume();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
        ? { mimeType: 'audio/webm;codecs=opus' } 
        : { mimeType: 'audio/webm' };
      
      const recorder = new MediaRecorder(stream, options);
      
      recorder.ondataavailable = (e) => {
        if (e.data.size === 0) return;
        const socket = socketRef.current;
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(e.data);
        }
      };

      recorder.onstop = () => {
        const socket = socketRef.current;
        if (socket?.readyState === WebSocket.OPEN) {
            socket.send('CMD:COMMIT_AUDIO');
            setIsProcessing(true);
        }
        stream.getTracks().forEach(track => track.stop());
        setIsRecording(false);
      };

      recorder.start(250); 
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      console.log("Recording started...");
    } catch (err) {
      console.error('Failed to access microphone:', err);
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (isRecording) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        // This will trigger the onstop event where CMD:COMMIT_AUDIO is sent
        mediaRecorderRef.current.stop();
        console.log("Recording stop requested...");
      }
    } else {
      // Interrupt logic for when AI is speaking or processing
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send('STOP');
      }
      audioQueueRef.current = [];
      isPlayingRef.current = false;
      if (activeSourceRef.current) {
        try {
          activeSourceRef.current.stop();
        } catch (err) {
          console.warn('Audio stop failed:', err);
        }
        activeSourceRef.current = null;
      }
      setAiIsSpeaking(false);
      setIsProcessing(false);
    }
  }, [isRecording]);

  const addMessage = useCallback((text: string, sender: 'user' | 'ai') => {
    setMessages(prev => [...prev, {
      id: Date.now().toString() + Math.random(),
      text,
      sender,
      timestamp: Date.now()
    }]);
  }, []);

  // --- CONNECTION ---
  const connect = useCallback((candidateIndex = 0) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    const url = socketCandidates[candidateIndex];
    if (!url) return;

    setStatus(ConnectionStatus.CONNECTING);
    const ws = new WebSocket(url);
    let opened = false;
    
    ws.onopen = () => {
      opened = true;
      setStatus(ConnectionStatus.OPEN);
      console.log('Neural Link Connected:', url);
      const targetSession = currentSessionIdRef.current || pendingSessionRef.current;
      if (targetSession) {
        ws.send(`SESSION:${targetSession}`);
      }
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        handleBinaryMessage(event.data);
        return;
      }
      const text = event.data;
      if (text === 'DONE') {
          setIsProcessing(false); // Ако сървърът приключи без аудио
      }
      else if (text.startsWith('SYS:SESSION:')) {
          const value = text.split('SYS:SESSION:', 2)[1];
          const parsed = value ? parseInt(value, 10) : NaN;
          if (!Number.isNaN(parsed)) {
              persistSessionId(parsed);
              pendingSessionRef.current = parsed;
              loadSessionWindow(parsed);
              refreshSessions();
          }
      }
      else if (text.startsWith('SYS:')) console.log(`%c[SYS] ${text}`, 'color: cyan');
      else if (text.startsWith('LOG:User:')) addMessage(text.replace('LOG:User:', '').trim(), 'user');
      else if (text.startsWith('LOG:AI:')) addMessage(text.replace('LOG:AI:', '').trim(), 'ai');
      else if (text.startsWith('RAG:')) {
          try {
              const data = JSON.parse(text.substring(4));
              console.log("📚 RAG Memory:", data);
          } catch(e) { console.error("RAG Parse Error", e); }
      }
    };

    ws.onclose = () => {
        setStatus(ConnectionStatus.CLOSED);
        setIsProcessing(false); // Reset on disconnect
        const nextIndex = (!opened && candidateIndex + 1 < socketCandidates.length) ? candidateIndex + 1 : 0;
        setTimeout(() => connect(nextIndex), opened ? 3000 : 500);
    };

    socketRef.current = ws;
  }, [addMessage, handleBinaryMessage, loadSessionWindow, persistSessionId, refreshSessions, socketCandidates]);

  const sendMessage = useCallback((text: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(text);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => { 
      socketRef.current?.close(); 
    };
  }, [connect]);

  return { 
    status, 
    messages, 
    sendMessage, 
    startRecording, 
    stopRecording,
    aiIsSpeaking,
    isRecording,
    isProcessing,
    sessions,
    currentSessionId,
    isLoadingSession,
    refreshSessions,
    switchSession,
    createNewSession,
  };
};
