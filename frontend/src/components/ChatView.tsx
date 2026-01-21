import React, { useEffect, useRef, useState, useLayoutEffect } from 'react';
import { ConnectionStatus, Message } from '../types';
import { IconMenu, IconSend, IconTerminal, IconWifi, IconWifiOff, IconWaveform } from './Icons';

interface ChatViewProps {
  status: ConnectionStatus;
  messages: Message[];
  onSendMessage: (text: string) => void;
  onCallStart: () => void;
  onOpenDrawer: () => void;
  sessionTitle?: string;
  isLoadingSession?: boolean;
}

// Inline Arrow Icon
const IconArrowDown: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="12" y1="5" x2="12" y2="19"></line>
    <polyline points="19 12 12 19 5 12"></polyline>
  </svg>
);

const ChatView: React.FC<ChatViewProps> = ({ status, messages, onSendMessage, onCallStart, onOpenDrawer, sessionTitle, isLoadingSession }) => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [inputValue, setInputValue] = useState('');
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isUserAtBottom, setIsUserAtBottom] = useState(true);

  // 1. SCROLL STATE TRACKING
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    
    // Threshold of 20px to consider "at bottom"
    const distanceToBottom = scrollHeight - scrollTop - clientHeight;
    const isBottom = distanceToBottom < 20;

    setIsUserAtBottom(isBottom);
    setShowScrollButton(!isBottom);
  };

  // 2. PASSIVE STICKY LOGIC
  // Only snaps if user is ALREADY at bottom.
  useLayoutEffect(() => {
    if (messages.length === 0) return;
    if (isUserAtBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, [messages, isUserAtBottom]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
      
      // Force scroll on user send
      setIsUserAtBottom(true);
      setTimeout(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 10);
    }
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setIsUserAtBottom(true);
  };

  const isOnline = status === ConnectionStatus.OPEN;

  return (
    // STRATEGY: NUCLEAR OPTION (Absolute Positioning)
    // The outer shell is FIXED to the viewport. No global scroll is physically possible.
    <div className="fixed inset-0 w-full h-[100dvh] bg-background overflow-hidden overscroll-none touch-none">
      
      {/* 1. HEADER - ABSOLUTE TOP */}
      {/* Height is hardcoded to h-16 (64px). It sits on top of everything (z-50). */}
      <header className="absolute top-0 left-0 right-0 h-16 z-50 flex items-center justify-between px-6 bg-background/95 backdrop-blur-xl border-b border-white/5 select-none touch-none">
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenDrawer}
            className="p-2 rounded-lg border border-white/5 bg-white/5 hover:bg-white/10 transition-colors"
            aria-label="Open sessions"
          >
            <IconMenu className="w-4 h-4 text-zinc-400" />
          </button>
          <div className="p-2 bg-white/5 rounded-lg border border-white/5">
            <IconTerminal className="w-4 h-4 text-accent" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-xs font-medium tracking-[0.2em] text-zinc-400">NEURAL LINK</h1>
            <div className="flex items-center gap-2 mt-0.5">
              <span className={`w-1 h-1 flex-none rounded-full ${isOnline ? 'bg-accent shadow-[0_0_8px_currentColor]' : 'bg-danger'}`}></span>
              <span className={`text-[10px] font-mono uppercase truncate ${isOnline ? 'text-accent' : 'text-danger'}`}>
                {isOnline ? 'Online' : 'Offline'}
              </span>
            </div>
            <div className="text-[10px] text-zinc-600 truncate max-w-[150px]">
              {isLoadingSession ? 'Loading...' : (sessionTitle || 'New Session')}
            </div>
          </div>
        </div>
        <div className="text-zinc-600 flex-none ml-2">
           {isOnline ? <IconWifi className="w-4 h-4 opacity-50" /> : <IconWifiOff className="w-4 h-4 text-danger/70" />}
        </div>
      </header>

      {/* 2. CHAT SCROLL ZONE - ANCHORED BETWEEN HEADER AND FOOTER */}
      {/* top-16 pushes it below header. bottom-[88px] pushes it above footer. */}
      {/* This is the ONLY element with overflow-y-auto. */}
      <div 
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="absolute top-16 bottom-[88px] left-0 right-0 overflow-y-auto overflow-x-hidden no-scrollbar px-4 py-6 z-10 overscroll-contain"
      >
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-700 space-y-4 opacity-50 select-none">
            <IconTerminal className="w-12 h-12 mb-2 opacity-20" />
            <p className="text-xs font-mono tracking-widest">AWAITING INPUT</p>
          </div>
        )}
        
        <div className="space-y-6">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={`flex w-full ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`max-w-[90%] text-sm leading-relaxed break-words ${
                  msg.sender === 'user' 
                    ? 'px-5 py-3 rounded-2xl bg-zinc-800/80 text-zinc-200 border border-white/10 rounded-tr-sm shadow-sm' 
                    : 'px-1 py-1 font-mono text-zinc-400 tracking-tight'
                }`}
              >
                {msg.sender === 'ai' && (
                  <span className="mr-2 text-accent opacity-50 select-none">{'>'}</span>
                )}
                {msg.text}
                {msg.sender === 'ai' && (
                   <span className="inline-block w-1.5 h-3 ml-1 bg-accent/50 animate-pulse align-middle" />
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} className="h-1" />
        </div>
      </div>

      {/* 3. FOOTER - ABSOLUTE BOTTOM */}
      <div className="absolute bottom-0 left-0 right-0 z-50 bg-background border-t border-white/5 pb-safe">
        
        {/* Toast Button (Floating relative to footer) */}
        <div className={`absolute -top-12 left-0 right-0 flex justify-center pointer-events-none transition-opacity duration-300 ${showScrollButton ? 'opacity-100' : 'opacity-0'}`}>
          <button 
            onClick={scrollToBottom}
            className="pointer-events-auto flex items-center gap-2 px-4 py-1.5 rounded-full bg-zinc-900/90 backdrop-blur-md border border-accent/20 text-accent shadow-lg shadow-black/50 hover:bg-zinc-800 transition-all transform hover:scale-105 active:scale-95"
          >
            <span className="text-[10px] font-mono tracking-widest uppercase">New Data</span>
            <IconArrowDown className="w-3 h-3 animate-bounce" />
          </button>
        </div>

        {/* Input Bar */}
        <div className="p-4">
          <form onSubmit={handleSubmit} className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={onCallStart}
              className="flex-none shrink-0 p-3 rounded-full text-zinc-500 hover:text-accent hover:bg-white/5 transition-colors touch-manipulation"
              aria-label="Start voice chat"
            >
              <IconWaveform className="w-4 h-4" />
            </button>

            <div className="flex-1 min-w-[12rem] flex items-center gap-2 bg-zinc-900/50 border border-white/5 rounded-full p-1 pl-4 backdrop-blur-md shadow-lg">
              <div className="flex-1 min-w-0">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Transmit message..."
                  className="w-full bg-transparent border-none text-sm text-zinc-200 focus:outline-none placeholder:text-zinc-700 font-mono py-2"
                />
              </div>

              <button
                type="submit"
                disabled={!inputValue.trim()}
                className={`flex-none shrink-0 p-3 rounded-full transition-all duration-300 touch-manipulation ${
                  inputValue.trim()
                    ? 'bg-zinc-800 text-zinc-200 hover:text-accent'
                    : 'bg-transparent text-zinc-700 cursor-not-allowed'
                }`}
                aria-label="Send message"
              >
                <IconSend className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>
      </div>

    </div>
  );
};

export default ChatView;
