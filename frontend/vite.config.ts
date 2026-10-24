import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'resolve-styles',
      resolveId(source, importer) {
        if (source.endsWith('.css') && importer) {
          const dir = path.dirname(importer);
          const localPath = path.resolve(dir, source);
          if (fs.existsSync(localPath)) {
            return null;
          }
          const fileName = path.basename(source);
          const stylesPath = path.resolve(__dirname, 'src/styles', fileName);
          if (fs.existsSync(stylesPath)) {
            return stylesPath;
          }
        }
        return null;
      },
    },
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@api': path.resolve(__dirname, './src/api'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@types': path.resolve(__dirname, './src/types'),
      '@styles': path.resolve(__dirname, './src/styles'),
    },
  },
});