import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { viteSingleFile } from 'vite-plugin-singlefile';
import staticMdPlugin from './vite-plugin-static-md.js';

export default defineConfig({
  base: './',
  plugins: [
    tailwindcss(),
    staticMdPlugin(),
    viteSingleFile(),
  ],
});
