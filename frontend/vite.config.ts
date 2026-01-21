import fs from 'fs';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const httpsConfig = (() => {
      const certDir = path.resolve(__dirname, '..', 'certs');
      try {
        return {
          key: fs.readFileSync(path.join(certDir, 'key.pem')),
          cert: fs.readFileSync(path.join(certDir, 'cert.pem')),
        };
      } catch (err) {
        console.warn('HTTPS dev certs not found; falling back to HTTP. Provide certs/key.pem and certs/cert.pem to enable HTTPS.');
        return false;
      }
    })();
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        https: httpsConfig,
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src'),
        }
      }
    };
});
