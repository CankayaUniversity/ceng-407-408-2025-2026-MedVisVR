from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import textwrap
from pathlib import Path


def _inline_format(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', text)
    return text


def markdown_to_html(md_text: str, title: str) -> str:
    lines = md_text.splitlines()
    body: list[str] = []
    in_ul = False
    in_ol = False
    in_code = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph).strip()
            if joined:
                body.append(f"<p>{_inline_format(joined)}</p>")
            paragraph = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            body.append("</ul>")
            in_ul = False
        if in_ol:
            body.append("</ol>")
            in_ol = False

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            flush_paragraph()
            close_lists()
            if not in_code:
                body.append("<pre><code>")
                in_code = True
            else:
                body.append("</code></pre>")
                in_code = False
            continue

        if in_code:
            body.append(html.escape(raw))
            continue

        if not line.strip():
            flush_paragraph()
            close_lists()
            continue

        if line.startswith("#"):
            flush_paragraph()
            close_lists()
            level = len(line) - len(line.lstrip("#"))
            level = max(1, min(level, 3))
            text = line[level:].strip()
            body.append(f"<h{level}>{_inline_format(text)}</h{level}>")
            continue

        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if not in_ol:
                body.append("<ol>")
                in_ol = True
            item = re.sub(r"^\d+\.\s+", "", line)
            body.append(f"<li>{_inline_format(item)}</li>")
            continue

        if line.startswith("- "):
            flush_paragraph()
            if in_ol:
                body.append("</ol>")
                in_ol = False
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{_inline_format(line[2:].strip())}</li>")
            continue

        paragraph.append(line)

    flush_paragraph()
    close_lists()

    css = """
    @page {
      size: A4;
      margin: 22mm 18mm 22mm 18mm;
    }
    body {
      font-family: "Georgia", "Times New Roman", serif;
      color: #111827;
      line-height: 1.5;
      font-size: 11.5pt;
    }
    .cover {
      margin-top: 50mm;
      page-break-after: always;
    }
    .cover h1 {
      font-size: 24pt;
      margin-bottom: 8pt;
    }
    .cover p {
      font-size: 12pt;
      color: #4b5563;
      margin: 4pt 0;
    }
    h1 {
      font-size: 20pt;
      border-bottom: 1px solid #d1d5db;
      padding-bottom: 6pt;
      margin-top: 0;
      margin-bottom: 14pt;
    }
    h2 {
      font-size: 15pt;
      margin-top: 18pt;
      margin-bottom: 8pt;
      color: #111827;
    }
    h3 {
      font-size: 12.5pt;
      margin-top: 14pt;
      margin-bottom: 6pt;
      color: #1f2937;
    }
    p {
      text-align: justify;
      margin: 0 0 9pt 0;
    }
    ul, ol {
      margin-top: 4pt;
      margin-bottom: 10pt;
      padding-left: 22pt;
    }
    li {
      margin-bottom: 4pt;
    }
    code {
      font-family: "Consolas", monospace;
      font-size: 9.5pt;
      background: #f3f4f6;
      padding: 1pt 3pt;
      border-radius: 3px;
    }
    pre {
      background: #f3f4f6;
      padding: 10pt;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      border: 1px solid #e5e7eb;
    }
    a {
      color: #1d4ed8;
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .footer {
      position: fixed;
      bottom: -6mm;
      left: 0;
      right: 0;
      font-size: 9pt;
      color: #6b7280;
      text-align: center;
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <section class="cover">
    <h1>{html.escape(title)}</h1>
    <p>Brain Tumor MRI Decision-Support Prototype</p>
    <p>Automatically generated from the project methodology document.</p>
  </section>
  {''.join(body)}
  <div class="footer">{html.escape(title)}</div>
</body>
</html>
"""


def render_pdf(html_path: Path, pdf_path: Path, browser_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    html_uri = html_path.resolve().as_uri()
    cmd = [
        str(browser_path),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--allow-file-access-from-files",
        "--print-to-pdf=" + str(pdf_path),
        html_uri,
    ]
    subprocess.run(cmd, check=True)


def _plain_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _markdown_blocks(md_text: str) -> list[tuple[str, str]]:
    lines = md_text.splitlines()
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(("p", " ".join(x.strip() for x in paragraph).strip()))
            paragraph = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            blocks.append(("code", "\n".join(code_lines)))
            code_lines = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(raw)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            level = max(1, min(level, 3))
            blocks.append((f"h{level}", line[level:].strip()))
            continue

        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            item = re.sub(r"^\d+\.\s+", "", line)
            blocks.append(("ol", item))
            continue

        if line.startswith("- "):
            flush_paragraph()
            blocks.append(("ul", line[2:].strip()))
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_code()
    return blocks


def render_pdf_with_matplotlib(md_text: str, pdf_path: Path, title: str) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    page_w = 8.27
    page_h = 11.69
    left = 0.08
    right = 0.92
    top = 0.94
    bottom = 0.06

    blocks = _markdown_blocks(md_text)
    wrapped: list[tuple[str, list[str]]] = []
    for kind, text in blocks:
        clean = _plain_inline(text)
        if kind == "h1":
            wrapped.append((kind, [clean]))
        elif kind == "h2":
            wrapped.append((kind, [clean]))
        elif kind == "h3":
            wrapped.append((kind, [clean]))
        elif kind == "ul":
            lines = textwrap.wrap(f"- {clean}", width=92, break_long_words=False, break_on_hyphens=False) or ["- "]
            wrapped.append((kind, lines))
        elif kind == "ol":
            lines = textwrap.wrap(clean, width=92, break_long_words=False, break_on_hyphens=False) or [clean]
            wrapped.append((kind, lines))
        elif kind == "code":
            lines = []
            for row in clean.splitlines():
                lines.extend(textwrap.wrap(row, width=88, break_long_words=True, break_on_hyphens=False) or [""])
            wrapped.append((kind, lines))
        else:
            lines = textwrap.wrap(clean, width=95, break_long_words=False, break_on_hyphens=False) or [clean]
            wrapped.append((kind, lines))

    def new_page():
        fig = plt.figure(figsize=(page_w, page_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        return fig, ax

    with PdfPages(str(pdf_path)) as pdf:
        fig, ax = new_page()
        ax.text(left, 0.78, title, fontsize=22, fontweight="bold", family="serif", va="top")
        ax.text(left, 0.72, "Brain Tumor MRI Decision-Support Prototype", fontsize=13, family="serif", va="top")
        ax.text(left, 0.68, "Automatically generated from the project methodology document.", fontsize=11, family="serif", va="top", color="#4b5563")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_page()
        y = top

        for kind, lines in wrapped:
            if kind == "h1":
                needed = 0.05
                if y - needed < bottom:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = new_page()
                    y = top
                ax.text(left, y, lines[0], fontsize=20, fontweight="bold", family="serif", va="top")
                y -= 0.045
                ax.plot([left, right], [y + 0.01, y + 0.01], color="#d1d5db", linewidth=1)
                y -= 0.02
                continue

            if kind == "h2":
                needed = 0.04
                if y - needed < bottom:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = new_page()
                    y = top
                ax.text(left, y, lines[0], fontsize=15, fontweight="bold", family="serif", va="top")
                y -= 0.032
                continue

            if kind == "h3":
                needed = 0.032
                if y - needed < bottom:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = new_page()
                    y = top
                ax.text(left, y, lines[0], fontsize=12.5, fontweight="bold", family="serif", va="top")
                y -= 0.026
                continue

            line_height = 0.018 if kind != "code" else 0.017
            gap_after = 0.01
            needed = len(lines) * line_height + gap_after
            if y - needed < bottom:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = new_page()
                y = top

            for idx, line in enumerate(lines):
                x = left
                if kind == "ol":
                    prefix = f"{idx + 1}. " if idx == 0 else "   "
                    text = prefix + line
                    family = "serif"
                    fontsize = 11
                elif kind == "code":
                    text = line
                    family = "monospace"
                    fontsize = 9.5
                else:
                    text = line
                    family = "serif"
                    fontsize = 11
                ax.text(x, y, text, fontsize=fontsize, family=family, va="top")
                y -= line_height
            y -= gap_after

        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--html-out", required=True)
    parser.add_argument("--pdf-out", required=True)
    parser.add_argument("--title", default="Methodology Report")
    parser.add_argument("--browser", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    html_path = Path(args.html_out)
    pdf_path = Path(args.pdf_out)
    browser_path = Path(args.browser)

    md_text = input_path.read_text(encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = markdown_to_html(md_text, args.title)
    html_path.write_text(html_text, encoding="utf-8")

    try:
        render_pdf(html_path.resolve(), pdf_path.resolve(), browser_path)
    except Exception:
        render_pdf_with_matplotlib(md_text, pdf_path.resolve(), args.title)

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError("PDF was not created.")

    print(pdf_path)


if __name__ == "__main__":
    main()
