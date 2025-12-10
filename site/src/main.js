import './style.css';
import { html, toc } from './writeup.md';

document.getElementById('content').innerHTML = html;

// Build TOC HTML from array of objects
const tocHtml = toc
  .map(item => {
    // Clean content - remove any HTML like anchor links
    const cleanContent = item.content.replace(/<[^>]*>/g, '').trim();
    const slug = cleanContent.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
    return `<a href="#${slug}" class="toc-level-${item.level}">${cleanContent}</a>`;
  })
  .join('');
document.getElementById('toc').innerHTML = tocHtml;
