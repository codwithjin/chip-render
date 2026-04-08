import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/process':  'http://localhost:8080',
      '/progress': 'http://localhost:8080',
      '/result':   'http://localhost:8080',
      '/phases':   'http://localhost:8080',
      '/api':      'http://localhost:8080',
      '/health':   'http://localhost:8080',
    }
  },
  build: {
    outDir: '../static_react',
    emptyOutDir: true,
  }
})
