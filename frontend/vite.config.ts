import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

// NOTE: Linter errors for missing modules are expected until dependencies are installed.
// This config will work once 'npm install' is run after Node.js is fixed.

// ESM-compatible __dirname
const __dirname = dirname(fileURLToPath(import.meta.url));

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    // Add proxy rules here to connect to MCP backend API endpoints
    // proxy: {
    //   '/api': 'http://localhost:8000',
    // },
  },
}); 