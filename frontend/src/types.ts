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
  files?: FileAttachment[];
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
  files?: FileAttachment[];
}

export interface ImageAttachment {
  id?: string;
  mime: string;
  data_b64: string;
  width?: number | null;
  height?: number | null;
  size_bytes?: number | null;
}

export interface FileAttachment {
  id: string;
  filename: string;
  mime?: string | null;
  size_bytes?: number | null;
  sha256?: string | null;
  indexed?: boolean;
  chunks?: number;
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
  sendMessage: (payload: { text: string; images?: ImageAttachment[]; files?: FileAttachment[] }) => void;
  uploadFile: (file: File) => Promise<FileAttachment>;
  connect: () => void;
  disconnect: () => void;
}
