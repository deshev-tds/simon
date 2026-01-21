import React from 'react';
import { SessionSummary } from '../types';
import { IconX, IconTerminal } from './Icons';

interface SessionDrawerProps {
  isOpen: boolean;
  sessions: SessionSummary[];
  currentSessionId?: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onClose: () => void;
  isLoading?: boolean;
}

const formatDate = (timestamp?: number) => {
  if (!timestamp) return '';
  return new Date(timestamp * 1000).toLocaleDateString();
};

const SessionDrawer: React.FC<SessionDrawerProps> = ({
  isOpen,
  sessions,
  currentSessionId,
  onSelect,
  onNew,
  onClose,
  isLoading,
}) => {
  return (
    <div className={`fixed inset-0 z-40 ${isOpen ? 'pointer-events-auto' : 'pointer-events-none'}`}>
      <div
        className={`absolute inset-0 bg-black/50 transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0'}`}
        onClick={onClose}
      />
      <div
        className={`absolute left-0 top-0 bottom-0 w-80 max-w-[82vw] bg-zinc-900/95 border-r border-white/5 backdrop-blur-xl shadow-2xl transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-4 py-4 border-b border-white/5">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200 uppercase tracking-[0.12em]">
            <IconTerminal className="w-4 h-4 text-accent" />
            Sessions
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-colors"
          >
            <IconX className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 border-b border-white/5">
          <button
            onClick={onNew}
            className="w-full py-2 rounded-lg bg-accent text-zinc-900 font-semibold text-sm hover:bg-accent-glow transition-colors"
          >
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2 max-h-[calc(100%-128px)]">
          {isLoading && (
            <div className="text-xs text-zinc-500">Loading sessions…</div>
          )}
          {!isLoading && sessions.length === 0 && (
            <div className="text-xs text-zinc-500">No sessions yet.</div>
          )}
          {sessions.map((session) => {
            const isActive = currentSessionId === session.id;
            const title = session.title?.trim() ? session.title : `Session ${session.id}`;
            return (
              <button
                key={session.id}
                onClick={() => onSelect(session.id)}
                className={`w-full text-left p-3 rounded-lg border transition-all duration-200 ${
                  isActive
                    ? 'border-accent/60 bg-accent-dim text-zinc-50'
                    : 'border-white/5 bg-zinc-900/60 text-zinc-300 hover:border-white/20'
                }`}
              >
                <div className="text-sm font-semibold truncate">{title}</div>
                {session.summary && (
                  <div className="text-xs text-zinc-500 truncate mt-1">
                    {session.summary}
                  </div>
                )}
                <div className="text-[10px] text-zinc-600 mt-1">
                  {formatDate(session.updated_at || session.created_at)}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SessionDrawer;
