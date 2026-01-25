import React from 'react';

type MarkdownMessageProps = {
  text: string;
  showCursor?: boolean;
};

type Block =
  | { type: 'hr' }
  | { type: 'heading'; level: number; text: string }
  | { type: 'quote'; text: string }
  | { type: 'list'; items: string[] }
  | { type: 'paragraph'; text: string };

const isHeading = (line: string) => /^\s*#{1,6}\s+/.test(line);
const isListItem = (line: string) => /^\s*[-*]\s+/.test(line);
const isQuote = (line: string) => /^\s*>\s?/.test(line);
const isRule = (line: string) => /^-{3,}$/.test(line.trim());

const renderInline = (text: string, keyPrefix: string): React.ReactNode[] => {
  const nodes: React.ReactNode[] = [];
  let i = 0;
  let keyIndex = 0;

  const nextKey = () => `${keyPrefix}-${keyIndex++}`;

  while (i < text.length) {
    const ch = text[i];
    if (ch === '\n') {
      nodes.push(<br key={nextKey()} />);
      i += 1;
      continue;
    }
    if (ch === '`') {
      const end = text.indexOf('`', i + 1);
      if (end !== -1) {
        const code = text.slice(i + 1, end);
        nodes.push(
          <code key={nextKey()} className="rounded bg-white/10 px-1 py-0.5 text-zinc-200">
            {code}
          </code>
        );
        i = end + 1;
        continue;
      }
    }
    if (text.startsWith('**', i) || text.startsWith('__', i)) {
      const marker = text.slice(i, i + 2);
      const end = text.indexOf(marker, i + 2);
      if (end !== -1) {
        const inner = text.slice(i + 2, end);
        const key = nextKey();
        nodes.push(
          <strong key={key} className="font-semibold text-zinc-200">
            {renderInline(inner, `${key}-b`)}
          </strong>
        );
        i = end + 2;
        continue;
      }
      nodes.push(marker);
      i += 2;
      continue;
    }
    if (ch === '*' || ch === '_') {
      const end = text.indexOf(ch, i + 1);
      if (end !== -1) {
        const inner = text.slice(i + 1, end);
        const key = nextKey();
        nodes.push(
          <em key={key} className="italic text-zinc-300">
            {renderInline(inner, `${key}-i`)}
          </em>
        );
        i = end + 1;
        continue;
      }
    }

    nodes.push(ch);
    i += 1;
  }

  return nodes;
};

const parseBlocks = (text: string): Block[] => {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\s---\s/g, '\n---\n');
  const lines = normalized.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    if (isRule(line)) {
      blocks.push({ type: 'hr' });
      i += 1;
      continue;
    }
    if (isHeading(line)) {
      const trimmed = line.trimStart();
      const [, hashes] = trimmed.match(/^(#{1,6})\s+/) || [];
      const level = hashes ? hashes.length : 3;
      const content = trimmed.slice(level + 1);
      blocks.push({ type: 'heading', level, text: content });
      i += 1;
      continue;
    }
    if (isQuote(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && isQuote(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ''));
        i += 1;
      }
      blocks.push({ type: 'quote', text: quoteLines.join('\n') });
      continue;
    }
    if (isListItem(line)) {
      const items: string[] = [];
      while (i < lines.length && isListItem(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
        i += 1;
      }
      blocks.push({ type: 'list', items });
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !isRule(lines[i]) &&
      !isHeading(lines[i]) &&
      !isQuote(lines[i]) &&
      !isListItem(lines[i])
    ) {
      paraLines.push(lines[i]);
      i += 1;
    }
    const paraText = paraLines.join('\n');
    blocks.push({ type: 'paragraph', text: paraText });
  }

  return blocks;
};

const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ text, showCursor = false }) => {
  const blocks = parseBlocks(text);
  const cursor = <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-accent/50 align-middle" />;

  if (blocks.length === 0 && showCursor) {
    return <div>{cursor}</div>;
  }

  return (
    <div className="space-y-3">
      {blocks.map((block, index) => {
        const isLast = showCursor && index === blocks.length - 1;
        if (block.type === 'hr') {
          return (
            <div key={`block-${index}`} className="space-y-3">
              <hr className="border-white/10" />
              {isLast && cursor}
            </div>
          );
        }
        if (block.type === 'heading') {
          const textNodes = renderInline(block.text, `block-${index}-h`);
          if (block.level === 1) {
            return (
              <h1 key={`block-${index}`} className="text-base font-semibold text-zinc-200">
                {textNodes}
                {isLast && cursor}
              </h1>
            );
          }
          if (block.level === 2) {
            return (
              <h2 key={`block-${index}`} className="text-sm font-semibold text-zinc-200">
                {textNodes}
                {isLast && cursor}
              </h2>
            );
          }
          return (
            <h3 key={`block-${index}`} className="text-sm font-semibold text-zinc-200">
              {textNodes}
              {isLast && cursor}
            </h3>
          );
        }
        if (block.type === 'quote') {
          return (
            <blockquote key={`block-${index}`} className="border-l border-white/10 pl-3 text-zinc-300">
              {renderInline(block.text, `block-${index}-q`)}
              {isLast && cursor}
            </blockquote>
          );
        }
        if (block.type === 'list') {
          return (
            <ul key={`block-${index}`} className="list-disc space-y-1 pl-5">
              {block.items.map((item, itemIndex) => {
                const isLastItem = isLast && itemIndex === block.items.length - 1;
                return (
                  <li key={`li-${index}-${itemIndex}`}>
                    {renderInline(item, `block-${index}-li${itemIndex}`)}
                    {isLastItem && cursor}
                  </li>
                );
              })}
            </ul>
          );
        }
        return (
          <p key={`block-${index}`} className="leading-relaxed">
            {renderInline(block.text, `block-${index}-p`)}
            {isLast && cursor}
          </p>
        );
      })}
    </div>
  );
};

export default MarkdownMessage;
