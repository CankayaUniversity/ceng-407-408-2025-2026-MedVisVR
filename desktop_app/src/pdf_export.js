const REPORT_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');

  *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family: 'Inter', Arial, sans-serif;
    font-size: 12pt;
    color: #1a202c;
    line-height: 1.75;
    background: #fff;
    padding: 0 2cm;
  }

  .watermark {
    font-size: 9pt; color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 10pt; margin-bottom: 24pt;
    display: flex; justify-content: space-between; align-items: center;
  }
  .watermark-brand { font-weight: 700; color: #1a6af0; font-size: 10pt; }

  h1 {
    font-family: 'Lora', Georgia, serif;
    font-size: 20pt; font-weight: 600; color: #0d1117;
    border-bottom: 2pt solid #e2e8f0;
    padding-bottom: 12pt; margin-bottom: 14pt; margin-top: 0; line-height: 1.3;
  }
  h2 { font-size: 13pt; font-weight: 700; color: #0d1117; margin-top: 24pt; margin-bottom: 8pt; }
  h3 { font-size: 10pt; font-weight: 700; color: #1a202c; margin-top: 16pt; margin-bottom: 6pt;
       text-transform: uppercase; letter-spacing: .05em; }

  p  { margin-bottom: 9pt; }
  ul, ol { padding-left: 20pt; margin-bottom: 10pt; }
  ul { list-style: disc; }
  ol { list-style: decimal; }
  li { margin-bottom: 4pt; }

  strong { font-weight: 700; color: #0d1117; }
  em     { color: #64748b; font-style: italic; }

  code {
    font-family: 'JetBrains Mono', monospace; font-size: 10pt;
    background: #eff6ff; color: #1e40af; padding: 1pt 5pt; border-radius: 3pt;
  }
  pre {
    font-family: 'JetBrains Mono', monospace; font-size: 10pt;
    background: #f1f5f9; color: #1a202c; padding: 12pt 16pt;
    border-radius: 4pt; border-left: 3pt solid #1a6af0;
    margin-bottom: 12pt; white-space: pre-wrap; word-break: break-word;
  }
  blockquote {
    border-left: 3pt solid #1a6af0; background: rgba(26,106,240,.04);
    color: #64748b; padding: 8pt 14pt; margin: 12pt 0;
    border-radius: 0 3pt 3pt 0; font-style: italic;
  }

  table { width: 100%; border-collapse: collapse; margin-bottom: 14pt; font-size: 11pt; page-break-inside: avoid; }
  th {
    background: #f1f5f9; color: #0d1117; font-weight: 700; padding: 7pt 12pt;
    border-bottom: 1.5pt solid #cbd5e1; text-align: left;
    font-size: 9.5pt; text-transform: uppercase; letter-spacing: .05em;
  }
  td { padding: 7pt 12pt; border-bottom: 1pt solid #e2e8f0; color: #1a202c; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: #fafafa; }

  hr { border: none; border-top: 1pt solid #e2e8f0; margin: 22pt 0; }

  [data-hl] {
    background: rgba(255,220,40,.38) !important;
    border-radius: 2pt; padding: 1pt 2pt;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }

  .user-notes { margin-top: 32pt; border-top: 2pt solid #e2e8f0; padding-top: 16pt; }
  .user-notes__title {
    font-size: 11pt; font-weight: 700; color: #0d1117;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10pt;
  }
  .user-notes__body {
    font-size: 11pt; color: #374151; line-height: 1.7;
    background: #fffbeb; border: 1pt solid #fde68a;
    border-radius: 4pt; padding: 12pt 14pt;
    white-space: pre-wrap; word-break: break-word;
  }

  .footer {
    margin-top: 32pt; padding-top: 12pt; border-top: 1pt solid #e2e8f0;
    font-size: 9pt; color: #94a3b8;
    display: flex; justify-content: space-between;
  }

  @page { margin: 2cm 2.2cm; size: A4 portrait; }
  h1,h2,h3 { page-break-after: avoid; }
  table,pre { page-break-inside: avoid; }

  .report-line { margin: 0 0 5pt; line-height: 1.55; }
  .report-line--metric {
    background: rgba(59,130,246,.10); border-left: 3pt solid #3b82f6;
    padding: 3pt 10pt; border-radius: 0 4pt 4pt 0;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .report-line--alert {
    background: rgba(239,68,68,.10); border-left: 3pt solid #ef4444;
    padding: 3pt 10pt; border-radius: 0 4pt 4pt 0;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .report-line--location {
    background: rgba(16,185,129,.10); border-left: 3pt solid #10b981;
    padding: 3pt 10pt; border-radius: 0 4pt 4pt 0;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .report-line--limitation {
    background: rgba(245,158,11,.08); border-left: 3pt solid #f59e0b;
    padding: 3pt 10pt; border-radius: 0 4pt 4pt 0;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .report-line-label { font-weight: 700; color: #0d1117; }
  .report-metric-value { color: #1a6af0; font-weight: 600; }
  .report-severity {
    display: inline-block; padding: 1pt 5pt; border-radius: 3pt;
    font-size: 9pt; font-weight: 700;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .report-severity.sev-critical { background:#ef4444; color:#fff; }
  .report-severity.sev-high     { background:#f97316; color:#fff; }
  .report-severity.sev-warn     { background:#f59e0b; color:#000; }
  .report-severity.sev-normal   { background:#10b981; color:#fff; }
  .report-heading--section {
    font-size: 10pt; font-weight: 700; text-transform: uppercase;
    letter-spacing: .8px; color: #1a6af0;
    background: rgba(26,106,240,.07); display: inline-block;
    padding: 2pt 8pt; border-radius: 3pt;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
`;

function markdownToHtml(md) {
  const lines = md.split('\n');
  const out = [];
  let inUl=false, inOl=false, inPre=false, preLines=[], inTable=false, tableRows=[];

  const closeList = () => {
    if (inUl) { out.push('</ul>'); inUl=false; }
    if (inOl) { out.push('</ol>'); inOl=false; }
  };
  const closeTable = () => {
    if (!inTable) return;
    if (tableRows.length > 0) {
      out.push('<table>');
      tableRows.forEach((row, i) => {
        const cells = row.split('|').map(c=>c.trim()).filter(c=>c);
        if (i === 0)
          out.push('<thead><tr>' + cells.map(c=>`<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>');
        else if (i === 1 && cells.every(c => /^[-:]+$/.test(c))) {}
        else
          out.push('<tr>' + cells.map(c=>`<td>${inline(c)}</td>`).join('') + '</tr>');
      });
      out.push('</tbody></table>');
    }
    inTable=false; tableRows=[];
  };
  const inline = t => t
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,         '<em>$1</em>')
    .replace(/`(.+?)`/g,           '<code>$1</code>');

  for (const line of lines) {
    if (line.startsWith('```')) {
      if (!inPre) { closeList(); closeTable(); inPre=true; preLines=[]; }
      else { out.push(`<pre>${preLines.join('\n').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre>`); inPre=false; }
      continue;
    }
    if (inPre) { preLines.push(line); continue; }
    if (line.startsWith('|')) { closeList(); inTable=true; tableRows.push(line); continue; }
    else if (inTable) closeTable();
    if (/^---+$/.test(line.trim())) { closeList(); out.push('<hr>'); continue; }
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) { closeList(); closeTable(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    if (line.startsWith('> ')) { closeList(); out.push(`<blockquote>${inline(line.slice(2))}</blockquote>`); continue; }
    const ul = line.match(/^[\-\*]\s+(.+)$/);
    if (ul) { closeTable(); if(!inUl){if(inOl){out.push('</ol>');inOl=false;}out.push('<ul>');inUl=true;} out.push(`<li>${inline(ul[1])}</li>`); continue; }
    const ol = line.match(/^\d+\.\s+(.+)$/);
    if (ol) { closeTable(); if(!inOl){if(inUl){out.push('</ul>');inUl=false;}out.push('<ol>');inOl=true;} out.push(`<li>${inline(ol[1])}</li>`); continue; }
    if (line.trim() === '') { closeList(); closeTable(); out.push(''); continue; }
    closeList(); closeTable(); out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  if (inPre) out.push(`<pre>${preLines.join('\n')}</pre>`);
  return out.join('\n');
}

function buildHtmlDocument(caseId, contentHtml, notes) {
  const dateStr = new Date().toLocaleDateString('en-US', { year:'numeric', month:'long', day:'numeric' });
  const notesHtml = notes?.trim()
    ? `<div class="user-notes">
         <div class="user-notes__title">📝 Clinical Notes</div>
         <div class="user-notes__body">${notes.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>
       </div>`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clinical Report — ${caseId}</title>
  <style>${REPORT_CSS}</style>
</head>
<body>
  <div class="watermark">
    <span class="watermark-brand">CARVIS Clinical AI</span>
    <span>Case: <strong>${caseId}</strong> &nbsp;·&nbsp; ${dateStr}</span>
  </div>
  ${contentHtml}
  ${notesHtml}
  <div class="footer">
    <span>CARVIS · Automated decision-support tool. Not a medical diagnosis.</span>
    <span>${dateStr}</span>
  </div>
</body>
</html>`;
}

function resolveContentHtml(markdownContent, reportDOMElement) {
  if (reportDOMElement) {
    const clone = reportDOMElement.cloneNode(true);
    clone.querySelectorAll('script').forEach(s => s.remove());
    return clone.innerHTML;
  }
  if (markdownContent) return markdownToHtml(markdownContent);
  return null;
}

export async function saveReportToFile(caseId, markdownContent, reportDOMElement, notes = '') {
  const contentHtml = resolveContentHtml(markdownContent, reportDOMElement);
  if (!contentHtml) {
    alert('Report not visible. Open the Report tab first.');
    return;
  }

  const html = buildHtmlDocument(caseId, contentHtml, notes);

  if (window.desktopRuntime?.savePdf) {
    const result = await window.desktopRuntime.savePdf(html, `CARVIS_${caseId}_Report.pdf`);
    if (!result.success && result.reason !== 'canceled') {
      alert(`PDF could not be saved: ${result.reason}`);
    }
    return;
  }

  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `CARVIS_${caseId}_Report.html`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
}

export async function downloadReportAsPdf(caseId, markdownContent, reportDOMElement, notes = '') {
  const contentHtml = resolveContentHtml(markdownContent, reportDOMElement);
  if (!contentHtml) {
    alert('Report not visible. Open the Report tab first.');
    return;
  }

  const html = buildHtmlDocument(caseId, contentHtml, notes);

  if (window.desktopRuntime?.printPdf) {
    await window.desktopRuntime.printPdf(html);
    return;
  }

  const win = window.open('', '_blank', 'width=960,height=750');
  if (!win) { alert('Pop-up blocked. Allow pop-ups and try again.'); return; }
  win.document.write(html);
  win.document.close();
  win.addEventListener('load', () => { setTimeout(() => { win.focus(); win.print(); }, 600); });
  setTimeout(() => { if (win.document.readyState === 'complete') { win.focus(); win.print(); } }, 1000);
}
