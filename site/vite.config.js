import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { plugin as mdPlugin, Mode } from 'vite-plugin-markdown';
import MarkdownIt from 'markdown-it';
import { katex } from '@mdit/plugin-katex';
import markdownItAnchor from 'markdown-it-anchor';

export default defineConfig({
  plugins: [
    tailwindcss(),
    mdPlugin({
      mode: [Mode.HTML, Mode.TOC],
      markdownIt: MarkdownIt({
        html: true,
        linkify: true,
        typographer: true,
      })
        .use(katex)
        .use(markdownItAnchor, {
          permalink: markdownItAnchor.permalink.linkInsideHeader({
            symbol: '#',
            placement: 'after',
          }),
          level: [1, 2, 3],
        }),
    }),
  ],
});
