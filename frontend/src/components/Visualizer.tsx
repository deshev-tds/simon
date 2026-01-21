import React, { useMemo } from 'react';

interface VisualizerProps {
  isRecording: boolean;   // User talking
  isProcessing: boolean;  // Transmitting
  isSpeaking: boolean;    // TTS playback
}

const Visualizer: React.FC<VisualizerProps> = ({ isRecording, isProcessing, isSpeaking }) => {
  const phase = useMemo(() => {
    if (isRecording) return 'recording';
    if (isProcessing) return 'processing';
    if (isSpeaking) return 'speaking';
    return 'idle';
  }, [isProcessing, isRecording, isSpeaking]);

  const haloStyle = useMemo(() => {
    const common = { animationTimingFunction: 'ease-in-out' as const, animationIterationCount: 'infinite' as const };
    let duration = '6s';
    let baseScale = 1.0;
    switch (phase) {
      case 'recording':
        duration = '2.2s'; baseScale = 1.02; break;
      case 'processing':
        duration = '3.6s'; baseScale = 1.08; break;
      case 'speaking':
        duration = '2.8s'; baseScale = 1.05; break;
      default:
        duration = '6s'; baseScale = 1.0; break;
    }
    return { 
      ...common, 
      animationName: 'glow-breathe', 
      animationDuration: duration, 
      transform: `scale(${baseScale})`,
      opacity: 0.6
    };
  }, [phase]);

  const rimStyle = useMemo(() => {
    const common = { animationTimingFunction: 'linear' as const, animationIterationCount: 'infinite' as const };
    return {
      ...common,
      animationName: 'slow-spin',
      animationDuration: phase === 'processing' ? '14s' : '20s',
      opacity: phase === 'idle' ? 0.3 : 0.6,
    };
  }, [phase]);

  return (
    <div className="relative flex items-center justify-center w-80 h-80">
      
      {/* LAYER 1: Halo */}
      <div 
        className="absolute inset-0 rounded-full bg-gradient-to-tr from-accent/0 via-accent/25 to-accent/0 blur-3xl transition-all duration-700"
        style={haloStyle}
      />

      {/* LAYER 2: Soft rim */}
      <div 
        className="absolute inset-8 rounded-full border border-accent/20 blur-sm transition-all duration-700"
        style={rimStyle}
      />

      {/* LAYER 4: Event Horizon */}
      <div 
        className={`absolute w-32 h-32 rounded-full bg-accent shadow-[0_0_50px_rgba(0,255,65,0.35)] transition-all duration-500 ease-out
        ${phase !== 'idle' ? 'scale-110 shadow-[0_0_80px_rgba(0,255,65,0.7)]' : 'scale-100 shadow-[0_0_30px_rgba(0,255,65,0.2)]'}`}
      >
        <div className="absolute inset-0 rounded-full bg-white/10 blur-md"></div>
      </div>

      {/* LAYER 5: Singularity */}
      <div 
        className={`relative z-10 w-32 h-32 rounded-full bg-black flex items-center justify-center overflow-hidden transition-transform duration-300
        ${phase !== 'idle' ? 'scale-95' : 'scale-100'}`}
        style={{
            boxShadow: 'inset 0 0 20px rgba(0,0,0,1), 0 0 0 1px rgba(255,255,255,0.08)' 
        }}
      >
         <div className={`absolute -top-10 -left-10 w-24 h-24 bg-gradient-to-br from-white/10 to-transparent rounded-full blur-xl transition-opacity duration-300 ${phase !== 'idle' ? 'opacity-0' : 'opacity-30'}`}></div>
      </div>
    </div>
  );
};

export default Visualizer;
