#!/usr/bin/env python3
"""
md2pdf_ollama.py — Convert a Markdown file to a styled PDF using a local Ollama model.

Pipeline:
    Markdown --> [Ollama / deepseek-r1:32b] --> print-ready HTML+CSS --> [renderer] --> PDF

The LLM does the typesetting (layout decisions, CSS, @page rules, table styling).
A deterministic renderer does the actual PDF generation, because an LLM cannot emit
binary PDF bytes.

Requirements:
    pip install requests
    # at least one renderer:
    pip install weasyprint          # preferred, pure-Python (needs `brew install pango` on macOS)
    # or have Google Chrome installed (headless fallback, zero extra setup on a Mac)
    # or: brew install wkhtmltopdf
    # optional deterministic fallback path:
    pip install markdown pygments

Usage:
    python3 md2pdf_ollama.py report.md
    python3 md2pdf_ollama.py report.md -o report.pdf --theme corporate
    python3 md2pdf_ollama.py report.md --keep-html --verbose
    python3 md2pdf_ollama.py report.md --no-llm          # skip the model, plain conversion
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")


DEFAULT_MODEL = "deepseek-r1:32b"
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

THEMES = {
    "clean": "Neutral serif body, generous whitespace, subtle grey rules. Reads like a "
             "well-set technical report.",
    "corporate": "Sans-serif throughout, a single restrained accent colour, tight table "
                 "styling with banded rows, suitable for a customer-facing deliverable.",
    "technical": "Compact sans-serif body, monospace code blocks with a light background, "
                 "dense tables, designed for architecture and runbook documents.",
    "academic": "Serif body at 11pt, justified paragraphs, numbered headings, footnote-style "
                "small print.",
}

SYSTEM_PROMPT = """You are a document typesetter. You convert Markdown into a single, \
complete, standalone HTML5 document intended for print-to-PDF rendering.

Hard rules:
- Output ONLY HTML. No commentary, no explanation, no Markdown code fences.
- Start with <!DOCTYPE html> and end with </html>.
- All CSS goes in one <style> block in <head>. No external stylesheets, no web fonts, no
  JavaScript, no network requests of any kind.
- Preserve ALL source content exactly: every heading, paragraph, list item, table row, code
  block and link. Do not summarise, reorder, rewrite, translate or add content.
- Convert Markdown structures faithfully: tables to <table>, fenced code to <pre><code>,
  blockquotes to <blockquote>, task lists to styled list items.
- Include @page rules: size, margins, and a footer page counter via @bottom-center where
  supported.
- Use page-break-inside: avoid on tables, figures and code blocks; page-break-after: avoid
  on headings.
- Use only system font stacks (e.g. -apple-system, "Helvetica Neue", Georgia, "SF Mono",
  Menlo, monospace).
"""

USER_TEMPLATE = """Typeset the Markdown document below as a print-ready HTML page.

Page size: {page_size}
Style direction: {theme_desc}
Document title: {title}

Return the complete HTML document and nothing else.

--- BEGIN MARKDOWN ---
{markdown}
--- END MARKDOWN ---
"""


# --------------------------------------------------------------------------- utilities

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  {msg}", file=sys.stderr)


def strip_reasoning(text: str) -> str:
    """deepseek-r1 emits a <think>...</think> chain-of-thought block. Remove it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Unterminated think block (truncated generation)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def strip_fences(text: str) -> str:
    """Remove ```html ... ``` wrappers the model may add despite instructions."""
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```$", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def extract_html(text: str) -> str:
    """Pull the HTML document out of a model response."""
    text = strip_fences(strip_reasoning(text))
    start = text.lower().find("<!doctype html")
    if start == -1:
        start = text.lower().find("<html")
    if start > 0:
        text = text[start:]
    end = text.lower().rfind("</html>")
    if end != -1:
        text = text[: end + len("</html>")]
    return text.strip()


def source_headings(markdown: str) -> list[str]:
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.M)]


def validate_html(html: str, markdown: str, verbose: bool) -> tuple[bool, str]:
    """Cheap sanity checks. LLMs silently drop content on long documents."""
    low = html.lower()
    if "<html" not in low or "</html>" not in low:
        return False, "response is not a complete HTML document"
    if "<style" not in low:
        return False, "no <style> block found"

    heads = source_headings(markdown)
    if heads:
        text_only = re.sub(r"<[^>]+>", " ", html)
        text_only = html_lib.unescape(text_only)
        norm = re.sub(r"\s+", " ", text_only).lower()
        missing = [h for h in heads if re.sub(r"\s+", " ", re.sub(r"[*_`\[\]]", "", h)).lower() not in norm]
        if len(missing) > len(heads) * 0.15:
            return False, f"{len(missing)}/{len(heads)} source headings missing from output"
        if missing:
            log(f"warning: headings not found in output: {missing[:5]}", verbose)

    src_rows = markdown.count("\n|")
    if src_rows > 4 and "<table" not in low:
        return False, "source has tables but output contains none"
    return True, "ok"


# ------------------------------------------------------------------------------ ollama

def ollama_alive(host: str) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=5).ok
    except requests.RequestException:
        return False


def model_present(host: str, model: str) -> bool:
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
    except (requests.RequestException, ValueError):
        return False
    names = {m.get("name", "") for m in tags}
    return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)


def call_ollama(host: str, model: str, system: str, user: str,
                num_ctx: int, temperature: float, timeout: int,
                verbose: bool) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": -1,
        },
    }
    t0 = time.time()
    log(f"calling {model} (num_ctx={num_ctx}, this can take a few minutes)...", verbose)
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    log(f"model returned {len(content):,} chars in {time.time() - t0:.1f}s "
        f"(eval {data.get('eval_count', '?')} tokens)", verbose)
    return content


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 512


# ------------------------------------------------------------- deterministic fallback

FALLBACK_CSS = """
@page { size: {page_size}; margin: 20mm 18mm; }
body { font: 11pt/1.55 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 16pt; margin-top: 1.6em; }
h3 { font-size: 13pt; }
h1, h2, h3, h4 { page-break-after: avoid; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 9.5pt;
       background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 4px; padding: 10px;
      overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eef1f4; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 14px; color: #555; }
img { max-width: 100%; }
a { color: #0a5aa8; text-decoration: none; }
"""


def fallback_html(markdown: str, title: str, page_size: str) -> str:
    try:
        import markdown as md_lib
    except ImportError:
        sys.exit("Fallback path needs: pip install markdown pygments")
    body = md_lib.markdown(
        markdown,
        extensions=["extra", "tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    css = FALLBACK_CSS.replace("{page_size}", page_size)
    return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html_lib.escape(title)}</title><style>{css}</style></head>"
            f"<body>\n{body}\n</body></html>")


# ---------------------------------------------------------------------------- renderer

def find_chrome() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for exe in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(exe)
        if found:
            candidates.insert(0, found)
    return next((c for c in candidates if os.path.exists(c)), None)


def render_pdf(html: str, out_pdf: Path, verbose: bool) -> str:
    """Try renderers in order of output quality. Returns the renderer used."""
    # 1. WeasyPrint — best CSS paged-media support (@page margin boxes, page counters)
    try:
        from weasyprint import HTML as WeasyHTML  # type: ignore
        log("rendering with WeasyPrint...", verbose)
        WeasyHTML(string=html, base_url=str(out_pdf.parent)).write_pdf(str(out_pdf))
        return "weasyprint"
    except ImportError:
        log("weasyprint not installed, trying next renderer", verbose)
    except Exception as exc:  # noqa: BLE001 - weasyprint raises many native errors
        log(f"weasyprint failed ({exc}), trying next renderer", verbose)

    # 2. Headless Chrome — already on most Macs, good fidelity, no page counters
    chrome = find_chrome()
    if chrome:
        log(f"rendering with headless Chrome ({Path(chrome).name})...", verbose)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "doc.html"
            src.write_text(html, encoding="utf-8")
            cmd = [
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", f"--user-data-dir={tmp}/profile",
                "--virtual-time-budget=5000", src.as_uri(),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if out_pdf.exists() and out_pdf.stat().st_size > 1000:
                return "chrome-headless"
            log(f"chrome failed: {proc.stderr[-400:]}", verbose)

    # 3. wkhtmltopdf
    wk = shutil.which("wkhtmltopdf")
    if wk:
        log("rendering with wkhtmltopdf...", verbose)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "doc.html"
            src.write_text(html, encoding="utf-8")
            proc = subprocess.run(
                [wk, "--enable-local-file-access", "--quiet", str(src), str(out_pdf)],
                capture_output=True, text=True, timeout=180,
            )
            if out_pdf.exists() and out_pdf.stat().st_size > 1000:
                return "wkhtmltopdf"
            log(f"wkhtmltopdf failed: {proc.stderr[-400:]}", verbose)

    sys.exit(
        "No working PDF renderer found. Install one of:\n"
        "  pip install weasyprint      (macOS also needs: brew install pango)\n"
        "  brew install --cask google-chrome\n"
        "  brew install wkhtmltopdf"
    )


# -------------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert Markdown to PDF, using a local Ollama model as the typesetter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="input .md file")
    p.add_argument("-o", "--output", type=Path, help="output .pdf (default: input name)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama base URL (default: {DEFAULT_HOST})")
    p.add_argument("--theme", choices=sorted(THEMES), default="clean", help="style direction")
    p.add_argument("--page-size", default="Letter", help="A4, Letter, Legal (default: Letter)")
    p.add_argument("--title", help="document title (default: first H1, else filename)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--num-ctx", type=int, default=0, help="context window; 0 = auto-size to document")
    p.add_argument("--timeout", type=int, default=1800, help="Ollama HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=2, help="regeneration attempts if validation fails")
    p.add_argument("--keep-html", action="store_true", help="also write the intermediate .html")
    p.add_argument("--html-only", action="store_true", help="stop after HTML, skip PDF rendering")
    p.add_argument("--no-llm", action="store_true", help="skip Ollama, use deterministic conversion")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if not args.input.is_file():
        sys.exit(f"Not found: {args.input}")
    markdown = args.input.read_text(encoding="utf-8")
    if not markdown.strip():
        sys.exit("Input file is empty.")

    heads = source_headings(markdown)
    title = args.title or (heads[0] if heads else args.input.stem)
    out_pdf = args.output or args.input.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    v = args.verbose

    print(f"Source : {args.input}  ({len(markdown):,} chars, {len(heads)} headings)")

    html = ""
    if args.no_llm:
        log("--no-llm: deterministic conversion", v)
        html = fallback_html(markdown, title, args.page_size)
    else:
        if not ollama_alive(args.host):
            sys.exit(f"Cannot reach Ollama at {args.host}. Start it with: ollama serve")
        if not model_present(args.host, args.model):
            sys.exit(f"Model '{args.model}' not found. Pull it with: ollama pull {args.model}")

        num_ctx = args.num_ctx or min(65536, max(8192, estimate_tokens(markdown) * 3))
        user_msg = USER_TEMPLATE.format(
            page_size=args.page_size,
            theme_desc=THEMES[args.theme],
            title=title,
            markdown=markdown,
        )

        for attempt in range(1, args.retries + 2):
            try:
                raw = call_ollama(args.host, args.model, SYSTEM_PROMPT, user_msg,
                                  num_ctx, args.temperature, args.timeout, v)
            except requests.RequestException as exc:
                sys.exit(f"Ollama request failed: {exc}")

            candidate = extract_html(raw)
            ok, reason = validate_html(candidate, markdown, v)
            if ok:
                html = candidate
                break
            print(f"  attempt {attempt}: rejected — {reason}", file=sys.stderr)
            user_msg += ("\n\nIMPORTANT: your previous attempt was rejected because "
                         f"{reason}. Return the COMPLETE document with all content intact.")

        if not html:
            print("  LLM output failed validation; falling back to deterministic conversion.",
                  file=sys.stderr)
            html = fallback_html(markdown, title, args.page_size)

    if args.keep_html or args.html_only:
        out_html = out_pdf.with_suffix(".html")
        out_html.write_text(html, encoding="utf-8")
        print(f"HTML   : {out_html}")
        if args.html_only:
            return 0

    renderer = render_pdf(html, out_pdf, v)
    size_kb = out_pdf.stat().st_size / 1024
    print(f"PDF    : {out_pdf}  ({size_kb:,.0f} KB, via {renderer})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
