import React, { useEffect, useRef, useState, useLayoutEffect } from 'react';
import { ConnectionStatus, FileAttachment, ImageAttachment, Message } from '../types';
import MarkdownMessage from './MarkdownMessage';
import { IconImage, IconMenu, IconPaperclip, IconSend, IconTerminal, IconWifi, IconWifiOff, IconWaveform, IconX } from './Icons';

interface ChatViewProps {
  status: ConnectionStatus;
  messages: Message[];
  onSendMessage: (payload: { text: string; images?: ImageAttachment[]; files?: FileAttachment[] }) => void;
  onUploadFile: (file: File) => Promise<FileAttachment>;
  onCallStart: () => void;
  onOpenDrawer: () => void;
  sessionTitle?: string;
  isLoadingSession?: boolean;
  isAwaitingResponse?: boolean;
}

// Inline Arrow Icon
const IconArrowDown: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="12" y1="5" x2="12" y2="19"></line>
    <polyline points="19 12 12 19 5 12"></polyline>
  </svg>
);

const ChatView: React.FC<ChatViewProps> = ({ status, messages, onSendMessage, onUploadFile, onCallStart, onOpenDrawer, sessionTitle, isLoadingSession, isAwaitingResponse }) => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const attachFileInputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState('');
  const [pendingImages, setPendingImages] = useState<(ImageAttachment & { previewUrl: string })[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const [pendingFiles, setPendingFiles] = useState<FileAttachment[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isUserAtBottom, setIsUserAtBottom] = useState(true);
  const hasStreamingMessage = messages.some((msg) => msg.sender === 'ai' && msg.isStreaming);
  const showIncomingIndicator = Boolean(isAwaitingResponse && !hasStreamingMessage);

  const MAX_IMAGES = 10;
  const MAX_IMAGE_MB = 8;
  const MAX_FILES = 5;
  const MAX_FILE_MB = 10;
  // Note: This is the processed image that is both:
  // 1) rendered in the chat history, and
  // 2) sent to the backend/model.
  //
  // The previous 1024px + JPEG-only pipeline was noticeably lossy for screenshots/text.
  // We keep images under the same size cap, but preserve PNG when possible and otherwise
  // search for the best JPEG quality that fits.
  const MAX_IMAGE_EDGE = 1536;
  const MIN_IMAGE_EDGE = 320;

  const approxB64Bytes = (b64: string) => Math.floor((b64.length * 3) / 4);

  const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

  const renderScaled = (img: HTMLImageElement, targetMaxEdge: number) => {
    const maxEdge = Math.max(img.width, img.height);
    const edge = Math.max(1, Math.round(targetMaxEdge));
    const scale = maxEdge > edge ? edge / maxEdge : 1;
    const width = Math.max(1, Math.round(img.width * scale));
    const height = Math.max(1, Math.round(img.height * scale));

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.imageSmoothingEnabled = true;
    // @ts-ignore: imageSmoothingQuality exists in all modern browsers we target.
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, width, height);
    return { canvas, width, height };
  };

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

  const buildDataUrl = (mime: string, base64: string) => {
    if (base64.startsWith('data:')) return base64;
    return `data:${mime};base64,${base64}`;
  };

  const getFileDownloadUrl = (fileId: string) => {
    const base = import.meta.env.VITE_API_URL;
    if (base) return `${base}/v1/files/${fileId}`;
    return `/v1/files/${fileId}`;
  };

  const formatBytes = (bytes?: number | null) => {
    const b = typeof bytes === 'number' ? bytes : 0;
    if (!b) return '';
    const mb = b / (1024 * 1024);
    if (mb >= 1) return `${mb.toFixed(1)}MB`;
    const kb = b / 1024;
    return `${Math.max(1, Math.round(kb))}KB`;
  };

  const processFile = async (file: File): Promise<(ImageAttachment & { previewUrl: string }) | null> => {
    if (!file.type.startsWith('image/')) {
      setImageError('Unsupported file type.');
      return null;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.src = objectUrl;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => {
        URL.revokeObjectURL(objectUrl);
        resolve();
      };
      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        reject(new Error('Image load failed'));
      };
    });

    const sizeLimitBytes = MAX_IMAGE_MB * 1024 * 1024;
    const preferPng = file.type === 'image/png';

    const maxEdge = Math.max(img.width, img.height);
    const startEdge = Math.min(MAX_IMAGE_EDGE, maxEdge);
    const edgeAttempts: number[] = [];
    let edge = startEdge;
    while (edgeAttempts.length < 6) {
      edgeAttempts.push(Math.max(MIN_IMAGE_EDGE, Math.round(edge)));
      if (edge <= MIN_IMAGE_EDGE) break;
      edge = Math.max(MIN_IMAGE_EDGE, Math.round(edge * 0.85));
      if (edgeAttempts.includes(edge)) break;
    }

    let mime: string | null = null;
    let base64: string | null = null;
    let sizeBytes: number | null = null;
    let width: number | null = null;
    let height: number | null = null;

    for (const attemptEdge of edgeAttempts) {
      const rendered = renderScaled(img, attemptEdge);
      if (!rendered) {
        setImageError('Failed to process image.');
        return null;
      }

      // Prefer preserving PNG for screenshots/text if it fits in the size budget.
      if (preferPng) {
        const dataUrlPng = rendered.canvas.toDataURL('image/png');
        const b64Png = dataUrlPng.split(',')[1] || '';
        const bytesPng = approxB64Bytes(b64Png);
        if (bytesPng <= sizeLimitBytes) {
          mime = 'image/png';
          base64 = b64Png;
          sizeBytes = bytesPng;
          width = rendered.width;
          height = rendered.height;
          break;
        }
      }

      // Otherwise use JPEG and pick the highest quality that fits.
      const maxQ = 0.92;
      const minQ = 0.5;
      const dataUrlMax = rendered.canvas.toDataURL('image/jpeg', maxQ);
      const b64Max = dataUrlMax.split(',')[1] || '';
      const bytesMax = approxB64Bytes(b64Max);
      if (bytesMax <= sizeLimitBytes) {
        mime = 'image/jpeg';
        base64 = b64Max;
        sizeBytes = bytesMax;
        width = rendered.width;
        height = rendered.height;
        break;
      }

      // Binary search the best JPEG quality that fits.
      let lo = minQ;
      let hi = maxQ;
      let bestB64 = '';
      let bestBytes = 0;
      for (let i = 0; i < 7; i++) {
        const q = clamp((lo + hi) / 2, minQ, maxQ);
        const dataUrlMid = rendered.canvas.toDataURL('image/jpeg', q);
        const b64Mid = dataUrlMid.split(',')[1] || '';
        const bytesMid = approxB64Bytes(b64Mid);
        if (bytesMid <= sizeLimitBytes) {
          bestB64 = b64Mid;
          bestBytes = bytesMid;
          lo = q; // try higher quality
        } else {
          hi = q; // need lower quality
        }
      }
      if (bestBytes > 0 && bestBytes <= sizeLimitBytes) {
        mime = 'image/jpeg';
        base64 = bestB64;
        sizeBytes = bestBytes;
        width = rendered.width;
        height = rendered.height;
        break;
      }
    }

    if (!mime || !base64 || !sizeBytes || !width || !height) {
      setImageError(`Image too large after compression. (limit: ${MAX_IMAGE_MB}MB)`);
      return null;
    }

    return {
      id: `${Date.now()}-${Math.random()}`,
      mime,
      data_b64: base64,
      width,
      height,
      size_bytes: sizeBytes,
      previewUrl: buildDataUrl(mime, base64),
    };
  };

  const handleFilesSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    setImageError(null);
    const remaining = MAX_IMAGES - pendingImages.length;
    const slice = files.slice(0, remaining);
    if (files.length > remaining) {
      setImageError(`Only ${MAX_IMAGES} images allowed per message.`);
    }

    const results: (ImageAttachment & { previewUrl: string })[] = [];
    for (const file of slice) {
      try {
        const processed = await processFile(file);
        if (processed) results.push(processed);
      } catch (err) {
        setImageError('Failed to process an image.');
      }
    }
    if (results.length > 0) {
      setPendingImages(prev => [...prev, ...results].slice(0, MAX_IMAGES));
    }
    e.target.value = '';
  };

  const removePendingImage = (id: string) => {
    setPendingImages(prev => prev.filter(img => img.id !== id));
  };

  const removePendingFile = (id: string) => {
    setPendingFiles(prev => prev.filter(f => f.id !== id));
  };

  const handleAttachFilesSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    setFileError(null);
    const remaining = MAX_FILES - pendingFiles.length;
    const slice = files.slice(0, remaining);
    if (files.length > remaining) {
      setFileError(`Only ${MAX_FILES} files allowed per message.`);
    }

    setIsUploadingFiles(true);
    const uploaded: FileAttachment[] = [];
    try {
      for (const file of slice) {
        if (file.size > MAX_FILE_MB * 1024 * 1024) {
          setFileError(`File too large: ${file.name} (limit: ${MAX_FILE_MB}MB)`);
          continue;
        }
        try {
          const meta = await onUploadFile(file);
          if (meta?.id) uploaded.push(meta);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          setFileError(`Upload failed: ${file.name} (${msg})`);
        }
      }
    } finally {
      setIsUploadingFiles(false);
      if (uploaded.length > 0) {
        setPendingFiles(prev => [...prev, ...uploaded].slice(0, MAX_FILES));
      }
      e.target.value = '';
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isUploadingFiles) return;
    if (inputValue.trim() || pendingImages.length > 0 || pendingFiles.length > 0) {
      onSendMessage({
        text: inputValue,
        images: pendingImages.map(({ previewUrl, ...rest }) => rest),
        files: pendingFiles,
      });
      setInputValue('');
      setPendingImages([]);
      setPendingFiles([]);
      setImageError(null);
      setFileError(null);
      
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
        {messages.length === 0 && !showIncomingIndicator && (
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
                {msg.sender === 'ai' ? (
                  <div className="flex gap-2">
                    <span className="text-accent opacity-50 select-none">{'>'}</span>
                    <div className="min-w-0 flex-1">
                      <MarkdownMessage text={msg.text} showCursor={msg.isStreaming} />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {msg.images && msg.images.length > 0 && (
                      <div className="grid grid-cols-2 gap-2">
                        {msg.images.map((img, idx) => (
                          <div key={`${msg.id}-img-${idx}`} className="overflow-hidden rounded-lg border border-white/10 bg-zinc-900/60">
                            <img
                              src={buildDataUrl(img.mime, img.data_b64)}
                              alt="attachment"
                              className="w-full h-full object-cover max-h-48"
                              loading="lazy"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.files && msg.files.length > 0 && (
                      <div className="space-y-1">
                        {msg.files.map((f) => (
                          <a
                            key={`${msg.id}-file-${f.id}`}
                            href={getFileDownloadUrl(f.id)}
                            className="block text-[11px] font-mono text-zinc-300 hover:text-white underline decoration-white/10 hover:decoration-accent/50 truncate"
                            title={f.filename}
                          >
                            {f.filename}{f.size_bytes ? ` (${formatBytes(f.size_bytes)})` : ''}
                          </a>
                        ))}
                      </div>
                    )}
                    {msg.text && <span className="whitespace-pre-wrap">{msg.text}</span>}
                  </div>
                )}
              </div>
            </div>
          ))}
          {showIncomingIndicator && (
            <div className="flex w-full justify-start">
              <div className="px-1 py-1 font-mono text-zinc-500 tracking-tight flex items-center gap-2">
                <span className="text-accent opacity-50 select-none">{'>'}</span>
                <span className="inline-block w-1.5 h-3 bg-accent/50 animate-pulse align-middle" />
                <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-600">Incoming</span>
              </div>
            </div>
          )}
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
        <div className="p-4 space-y-3">
          {pendingImages.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {pendingImages.map((img) => (
                <div key={img.id} className="relative w-20 h-20 rounded-lg overflow-hidden border border-white/10 bg-zinc-900/70">
                  <img src={img.previewUrl} alt="pending upload" className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => removePendingImage(String(img.id))}
                    className="absolute top-1 right-1 p-1 rounded-full bg-zinc-900/80 text-zinc-300 hover:text-white border border-white/10"
                    aria-label="Remove image"
                  >
                    <IconX className="w-3 h-3" />
                  </button>
                </div>
              ))}
              <div className="flex items-center text-[10px] text-zinc-500 font-mono tracking-widest px-2">
                {pendingImages.length} / {MAX_IMAGES}
              </div>
            </div>
          )}
          {pendingFiles.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {pendingFiles.map((f) => (
                <div key={f.id} className="flex items-center gap-2 max-w-full px-3 py-2 rounded-full border border-white/10 bg-zinc-900/70">
                  <span className="text-[10px] font-mono text-zinc-300 truncate max-w-[14rem]" title={f.filename}>
                    {f.filename}
                  </span>
                  {f.size_bytes ? (
                    <span className="text-[10px] font-mono text-zinc-600 flex-none">{formatBytes(f.size_bytes)}</span>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => removePendingFile(f.id)}
                    className="p-1 rounded-full bg-zinc-900/80 text-zinc-300 hover:text-white border border-white/10 flex-none"
                    aria-label="Remove file"
                  >
                    <IconX className="w-3 h-3" />
                  </button>
                </div>
              ))}
              {isUploadingFiles && (
                <div className="flex items-center text-[10px] text-zinc-500 font-mono tracking-widest px-2">
                  Uploading…
                </div>
              )}
              <div className="flex items-center text-[10px] text-zinc-500 font-mono tracking-widest px-2">
                {pendingFiles.length} / {MAX_FILES}
              </div>
            </div>
          )}
          {imageError && (
            <div className="text-[10px] text-danger/80 font-mono tracking-widest uppercase">
              {imageError}
            </div>
          )}
          {fileError && (
            <div className="text-[10px] text-danger/80 font-mono tracking-widest uppercase">
              {fileError}
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={onCallStart}
              className="flex-none shrink-0 p-3 rounded-full text-zinc-500 hover:text-accent hover:bg-white/5 transition-colors touch-manipulation"
              aria-label="Start voice chat"
            >
              <IconWaveform className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => imageInputRef.current?.click()}
              className="flex-none shrink-0 p-3 rounded-full text-zinc-500 hover:text-accent hover:bg-white/5 transition-colors touch-manipulation"
              aria-label="Attach images"
            >
              <IconImage className="w-4 h-4" />
            </button>

            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleFilesSelected}
            />

            <button
              type="button"
              onClick={() => attachFileInputRef.current?.click()}
              className="flex-none shrink-0 p-3 rounded-full text-zinc-500 hover:text-accent hover:bg-white/5 transition-colors touch-manipulation"
              aria-label="Attach files"
            >
              <IconPaperclip className="w-4 h-4" />
            </button>

            <input
              ref={attachFileInputRef}
              type="file"
              accept=".txt,.md,.py,.js,.ts,.json,.yaml,.yml,.toml,.ini,.cfg,.log,.csv,application/pdf"
              multiple
              className="hidden"
              onChange={handleAttachFilesSelected}
            />

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
                disabled={isUploadingFiles || (!inputValue.trim() && pendingImages.length === 0 && pendingFiles.length === 0)}
                className={`flex-none shrink-0 p-3 rounded-full transition-all duration-300 touch-manipulation ${
                  (inputValue.trim() || pendingImages.length > 0 || pendingFiles.length > 0) && !isUploadingFiles
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
