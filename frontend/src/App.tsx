import React, { useState } from 'react';
import ChatView from './components/ChatView';
import VoiceOverlay from './components/VoiceOverlay';
import { useNeuralSocket } from './hooks/useNeuralSocket';
import { AppMode } from './types';
import { IconWaveform } from './components/Icons';
import SessionDrawer from './components/SessionDrawer';

const App: React.FC = () => {
  const { 
    status, 
    messages, 
    sendMessage, 
    uploadFile,
    startRecording, 
    stopRecording, 
    aiIsSpeaking,
    isRecording,
    isProcessing,
    isAwaitingResponse,
    liveTranscript,
    sessions,
    currentSessionId,
    isLoadingSession,
    switchSession,
    createNewSession,
  } = useNeuralSocket();

  const [mode, setMode] = useState<AppMode>(AppMode.CHAT);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleCloseVoice = () => {
    stopRecording(); 
    setMode(AppMode.CHAT);
  };

  const currentSessionTitle = sessions.find((s) => s.id === currentSessionId)?.title || 'New Session';

  return (
    <div className="relative w-full h-screen overflow-hidden bg-background text-text font-sans">
      <SessionDrawer
        isOpen={drawerOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelect={(id) => { switchSession(id); setDrawerOpen(false); }}
        onNew={() => { createNewSession(); setDrawerOpen(false); }}
        onClose={() => setDrawerOpen(false)}
        isLoading={isLoadingSession}
      />

      <ChatView 
        status={status}
        messages={messages}
        onSendMessage={sendMessage}
        onUploadFile={uploadFile}
        onCallStart={() => setMode(AppMode.VOICE)}
        onOpenDrawer={() => setDrawerOpen(true)}
        sessionTitle={currentSessionTitle}
        isLoadingSession={isLoadingSession}
        isAwaitingResponse={isAwaitingResponse}
      />

      <div className={`absolute bottom-20 right-6 md:bottom-10 z-20 transition-transform duration-300 ${mode === AppMode.VOICE ? 'translate-y-32' : 'translate-y-0'}`}>
        <button
          onClick={() => setMode(AppMode.VOICE)}
          className="group relative flex items-center justify-center w-16 h-16 rounded-full bg-accent hover:bg-accent-glow text-zinc-900 shadow-lg shadow-accent/20 transition-all duration-300 hover:scale-105 active:scale-95"
        >
          <div className="absolute inset-0 rounded-full border border-white/20 animate-pulse-slow"></div>
          <IconWaveform className="w-8 h-8" />
        </button>
      </div>

      <VoiceOverlay 
        isOpen={mode === AppMode.VOICE} 
        onClose={handleCloseVoice}
        startRecording={startRecording}
        stopRecording={stopRecording}
        aiIsSpeaking={aiIsSpeaking}
        isRecording={isRecording}
        isProcessing={isProcessing}
        liveTranscript={liveTranscript}
      />
    </div>
  );
};

export default App;
