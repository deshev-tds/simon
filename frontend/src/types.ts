export enum ConnectionStatus {
  CONNECTING = 'CONNECTING',
  OPEN = 'OPEN',
  CLOSED = 'CLOSED',
  ERROR = 'ERROR'
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: number;
  isSystem?: boolean;
  isStreaming?: boolean;
  images?: ImageAttachment[];
}

export enum AppMode {
  CHAT = 'CHAT',
  VOICE = 'VOICE'
}

export interface SessionSummary {
  id: number;
  title: string;
  summary?: string;
  tags?: string;
  model?: string | null;
  created_at: number;
  updated_at: number;
}

export interface StoredMessage {
  id: number | string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at?: number;
  tokens?: number;
  attachments?: ImageAttachment[];
}

export interface ImageAttachment {
  id?: string;
  mime: string;
  data_b64: string;
  width?: number | null;
  height?: number | null;
  size_bytes?: number | null;
}

export interface LiveTranscript {
  stable: string;
  draft: string;
  isFinal?: boolean;
}

export interface SessionWindow {
  session: SessionSummary;
  anchors: StoredMessage[];
  recents: StoredMessage[];
}

export interface NeuralSocketHook {
  status: ConnectionStatus;
  messages: Message[];
  sendMessage: (payload: { text: string; images?: ImageAttachment[] }) => void;
  connect: () => void;
  disconnect: () => void;
}
