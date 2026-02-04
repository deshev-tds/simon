import React, { useState, useEffect } from 'react';
import { IconX, IconMic, IconMicOff } from './Icons';
import Visualizer from './Visualizer';
import { LiveTranscript } from '../types';

interface VoiceOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  startRecording: () => void;
  stopRecording: () => void;
  aiIsSpeaking: boolean;
  isRecording: boolean;
  isProcessing: boolean;
  liveTranscript: LiveTranscript | null;
}

const VoiceOverlay: React.FC<VoiceOverlayProps> = ({ 
  isOpen, 
  onClose, 
  startRecording, 
  stopRecording, 
  aiIsSpeaking,
  isRecording,
  isProcessing,
  liveTranscript
}) => {
  const [isMuted, setIsMuted] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setIsVisible(true);
    } else {
      const timer = setTimeout(() => setIsVisible(false), 500);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const handleInteraction = () => {
    if (aiIsSpeaking || isProcessing) {
      // Interrupt (both speaking and thinking)
      stopRecording(); 
    } else if (isRecording) {
      // Send
      stopRecording();
    } else {
      // Start
      if (!isMuted) startRecording();
    }
  };

  if (!isVisible && !isOpen) return null;

  // --- STATUS LOGIC ---
  let statusText = "TAP TO SPEAK";
  let statusColor = "bg-zinc-600";
  let textColor = "text-zinc-500";
  let isPulse = false;

  if (aiIsSpeaking) {
      statusText = "RECEIVING TRANSMISSION";
      statusColor = "bg-accent";
      textColor = "text-accent";
      isPulse = true;
  } else if (isProcessing) {
      statusText = "PROCESSING UPLINK...";
      statusColor = "bg-white"; // White/Zinc pulse for thinking
      textColor = "text-zinc-300";
      isPulse = true;
  } else if (isRecording) {
      statusText = "LISTENING...";
      statusColor = "bg-danger"; // Red for recording (optional, or keep accent)
      textColor = "text-danger";
      isPulse = true;
  }

  const stableText = liveTranscript?.stable?.trim() ?? "";
  const draftText = liveTranscript?.draft?.trim() ?? "";
  const showTranscript = Boolean(stableText || draftText);

  return (
    <div 
      className={`fixed inset-0 z-50 flex flex-col items-center justify-between bg-zinc-950/98 backdrop-blur-3xl transition-opacity duration-500 ease-in-out ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
    >
      
      {/* Top Status */}
      <div className="pt-16 text-center space-y-2 pointer-events-none">
        <h2 className="text-zinc-500 text-[10px] font-mono tracking-[0.3em] uppercase">Neural Uplink</h2>
        <div className="flex items-center justify-center gap-2">
           <span className={`w-1.5 h-1.5 rounded-full ${statusColor} ${isPulse ? 'animate-pulse' : ''} shadow-[0_0_10px_currentColor]`}></span>
           <span className={`${textColor} text-xs font-semibold tracking-wider transition-colors duration-300`}>
             {statusText}
           </span>
        </div>
      </div>

      {showTranscript && (
        <div className="w-full max-w-3xl px-6 -mt-4">
          <div className={`rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md px-5 py-4 shadow-lg ${liveTranscript?.isFinal ? 'shadow-accent/20 border-accent/30' : ''}`}>
            <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-zinc-500 mb-2">
              {liveTranscript?.isFinal ? 'Final Transcript' : 'Live Transcript'}
            </div>
            <div className="text-sm leading-relaxed font-mono text-zinc-200 whitespace-pre-wrap">
              <span className="text-zinc-200">{stableText}</span>
              {draftText && (
                <span className="text-zinc-500 italic">{stableText ? ' ' : ''}{draftText}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main Interactive Visualizer */}
      <div className="flex-1 flex items-center justify-center w-full">
         <button 
            onClick={handleInteraction}
            className="relative group outline-none focus:outline-none"
            style={{ WebkitTapHighlightColor: 'transparent' }}
         >
            {/* Visualizer active during Speak, Rec OR Processing */}
          <Visualizer 
            isRecording={isRecording} 
            isProcessing={isProcessing} 
            isSpeaking={aiIsSpeaking} 
          />
            
            {/* Hint for interaction */}
            {!aiIsSpeaking && !isRecording && !isProcessing && (
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
                    <span className="text-[10px] tracking-widest text-zinc-500 font-mono">INITIALIZE</span>
                </div>
            )}
         </button>
      </div>

      {/* Bottom Controls */}
      <div className="pb-16 w-full max-w-xs px-8">
        <div className="flex items-center justify-between">
          
          <button 
            onClick={() => setIsMuted(!isMuted)}
            className={`p-5 rounded-full transition-all duration-300 ${isMuted ? 'bg-zinc-200 text-zinc-900' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-200 border border-white/5'}`}
          >
            {isMuted ? <IconMicOff className="w-5 h-5" /> : <IconMic className="w-5 h-5" />}
          </button>

          <button 
            onClick={onClose}
            className="p-6 rounded-full bg-danger/5 text-danger hover:bg-danger/10 hover:shadow-[0_0_20px_rgba(190,18,60,0.2)] transition-all duration-300 transform border border-danger/20 group"
          >
            <IconX className="w-6 h-6 transition-transform duration-500 group-hover:rotate-90 group-hover:scale-110" />
          </button>

          <div className="w-16"></div> 
        </div>
      </div>
    </div>
  );
};

export default VoiceOverlay;
