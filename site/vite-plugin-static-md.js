import fs from 'fs';
import MarkdownIt from 'markdown-it';
import { katex } from '@mdit/plugin-katex';
import markdownItAnchor from 'markdown-it-anchor';

export default function staticMdPlugin() {
  let config;

  return {
    name: 'static-md',
    configResolved(resolvedConfig) {
      config = resolvedConfig;
    },
    transformIndexHtml(html) {
      // Only run during build
      if (config.command !== 'build') return html;

      // Read and render markdown
      const markdown = fs.readFileSync('./src/writeup.md', 'utf-8');
      const md = MarkdownIt({
        html: true,
        linkify: true,
        typographer: true,
      })
        .use(katex)
        .use(markdownItAnchor, {
          permalink: markdownItAnchor.permalink.headerLink({
            symbol: '#',
            safariReaderFix: true,
          }),
          level: [1, 2, 3],
        });

      const content = md.render(markdown);

      // Extract TOC
      const tokens = md.parse(markdown, {});
      const toc = [];
      tokens.forEach((token, idx) => {
        if (token.type === 'heading_open') {
          const level = parseInt(token.tag.slice(1));
          const contentToken = tokens[idx + 1];
          if (contentToken && contentToken.type === 'inline' && level <= 3) {
            // Use children for text content
            let cleanContent = '';
            if (contentToken.children) {
              cleanContent = contentToken.children
                .filter(t => t.type === 'text')
                .map(t => t.content)
                .join(' ')
                .trim();
            }
            if (cleanContent) {
              toc.push({ level, content: cleanContent });
            }
          }
        }
      });

      const tocHtml = toc
        .map((item) => {
          const slug = item.content.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
          return `<a href="#${slug}" class="toc-level-${item.level}">${item.content}</a>`;
        })
        .join('\n');

      // Inject into HTML and remove script tag
      return html
        .replace('<aside id="toc"></aside>', `<aside id="toc">${tocHtml}</aside>`)
        .replace('<main id="content"></main>', `<main id="content">${content}</main>`)
        .replace(/<script[^>]*src="\/src\/main\.js"[^>]*><\/script>/, '');
    },
  };
}
