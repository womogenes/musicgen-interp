import './style.css';
import { html, toc } from './writeup.md';

document.getElementById('content').innerHTML = html;

// Build TOC HTML from array of objects
const tocHtml = toc
  .map(item => {
    const slug = item.content.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
    return `<a href="#${slug}" class="toc-level-${item.level}">${item.content}</a>`;
  })
  .join('');
document.getElementById('toc').innerHTML = tocHtml;
