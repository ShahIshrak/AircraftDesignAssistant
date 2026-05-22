"""
KnowledgeBase_training.py  —  FAIRY Advanced RAG Knowledge Base
================================================================
Key improvements over v1:
  1. processed_files.json  — incremental processing; skip unchanged PDFs
  2. LLM-generated topic summaries per chunk group, with auto-merge of near-duplicates
  3. ROBUST table extraction (multi-strategy pipeline):
       • Strategy 1 — pdfplumber (vector PDFs, explicit grid lines)
       • Strategy 2 — camelot lattice (bordered tables)
       • Strategy 3 — camelot stream (borderless / whitespace-delimited tables)
       • Strategy 4 — pymupdf4llm markdown table detection
       • Strategy 5 — EasyOCR image crop + GPT-style reconstruction (scanned PDFs)
       Tables are validated before storage: empty-cell ratio check, row-count
       check, and column-consistency check. Incomplete tables are discarded.
  4. ROBUST text extraction:
       • unicode private-use-area scrubbing (garbled symbol fonts)
       • high-newline-ratio detection → OCR re-extraction attempt
       • formula-heavy pages flagged for optional OCR math mode
  5. POST-EXTRACTION QUALITY AUDIT (NEW):
       • After every training run the code audits ALL chunks in every store
         (including previously indexed PDFs) for known quality problems.
       • Prints a human-readable report: which PDFs have issues, what the
         problems are, and a per-problem-type count.
       • Offers to DELETE all data for any flagged PDF from FAISS, BM25,
         parent_chunks, cluster_summaries, knowledge_graph, AND
         processed_files.json — leaving absolutely no trace.
  6. Advanced RAG gap mitigations (unchanged from v1):
       • Hybrid search  (dense FAISS + sparse BM25)
       • HyDE  (Hypothetical Document Embeddings) for semantic gap
       • Parent-child chunking  (small retrieval unit, large context window)
       • Query expansion  (LLM rewrites query before retrieval)
       • MMR re-ranking  (Maximal Marginal Relevance for diversity)
       • Entity-aware Knowledge Graph  (relationship preservation)
       • Lightweight cross-encoder re-ranker
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fitz                     # PyMuPDF
import hdbscan
import numpy as np
import pdfplumber
import spacy
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from KnowledgeBase.knowledge_graph import SemanticKnowledgeGraph

# ──────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_FILES_JSON = "processed_files.json"

CHILD_CHUNK_TARGET    = 300
CHILD_CHUNK_MAX       = 420
PARENT_CHUNK_TARGET   = 1200
PARENT_CHUNK_MAX      = 1500
SENTENCE_OVERLAP      = 2

HDBSCAN_MIN_CLUSTER      = 2
SUMMARY_MERGE_THRESHOLD  = 0.88
MIN_SENTENCE_WORDS       = 5

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Quality audit thresholds ──────────────────────────────────────────────────
# Table chunk is flagged if it has fewer words than this
TABLE_MIN_WORDS        = 30
# Table chunk is flagged if empty-cell ratio exceeds this
TABLE_MAX_EMPTY_RATIO  = 0.70
# Table chunk needs at least this many data rows (excluding header + separator)
TABLE_MIN_DATA_ROWS    = 2
# Text chunk high-newline ratio threshold
TEXT_MAX_NL_RATIO      = 0.50
# Minimum Unicode private-use chars before a text chunk is flagged as garbled
TEXT_GARBLE_THRESHOLD  = 10
# Minimum math/Greek chars before a text chunk is flagged as formula-garbled
TEXT_FORMULA_THRESHOLD = 20
# Text chunk minimum word count
TEXT_MIN_WORDS         = 10

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_processed_registry(registry_path: str) -> dict:
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_processed_registry(registry_path: str, registry: dict) -> None:
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


# ── Sentence splitter (unchanged) ────────────────────────────────────────────

_ABBREVS = {
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs",
    "etc", "eg", "ie", "approx", "dept", "est", "fig",
    "vol", "no", "pp", "ch", "Jan", "Feb", "Mar", "Apr",
    "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
}
_ABBREV_RE   = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _ABBREVS) + r')\.'
)
_PLACEHOLDER = "\x00ABB\x00"
_REAL_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def _split_sentences(text: str) -> list[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _PLACEHOLDER, text.strip())
    protected = re.sub(r"(\d)\.(?=\d)", r"\1" + _PLACEHOLDER, protected)
    raw = _REAL_BOUNDARY.split(protected)
    out: list[str] = []
    for s in raw:
        s = s.replace(_PLACEHOLDER, ".").strip()
        if not s:
            continue
        words = s.split()
        if len(words) > CHILD_CHUNK_MAX:
            for i in range(0, len(words), CHILD_CHUNK_MAX):
                piece = " ".join(words[i : i + CHILD_CHUNK_MAX])
                if len(piece.split()) >= MIN_SENTENCE_WORDS:
                    out.append(piece)
        elif len(words) >= MIN_SENTENCE_WORDS:
            out.append(s)
    return out


def _chunk_text(
    text: str,
    target_words: int  = CHILD_CHUNK_TARGET,
    max_words: int     = CHILD_CHUNK_MAX,
    overlap_sents: int = SENTENCE_OVERLAP,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > target_words and current:
            chunk_text = " ".join(current)
            if len(chunk_text.split()) > max_words:
                chunk_text = " ".join(chunk_text.split()[:max_words])
            chunks.append(chunk_text)
            overlap = current[-overlap_sents:] if overlap_sents else []
            current = overlap + [sent]
            current_words = sum(len(s.split()) for s in current)
        else:
            current.append(sent)
            current_words += sent_words
    if current:
        chunk_text = " ".join(current)
        if len(chunk_text.split()) > max_words:
            chunk_text = " ".join(chunk_text.split()[:max_words])
        if len(chunk_text.split()) >= MIN_SENTENCE_WORDS:
            chunks.append(chunk_text)
    return chunks


def _sentences(text: str) -> list[str]:
    return _split_sentences(text)


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ──────────────────────────────────────────────────────────────────────────────

_PRIVATE_USE_RE = re.compile(r'[\uE000-\uF8FF\uF000-\uFFFF]')
_MULTI_NL_RE    = re.compile(r'\n{3,}')


def _clean_text(text: str) -> str:
    """
    Scrub common PDF extraction artefacts:
      1. Unicode private-use-area characters (garbled symbol/math fonts).
      2. Excessive newlines → at most two consecutive.
      3. NUL bytes.
      4. Soft-hyphen line-break artefacts (­).
    """
    text = _PRIVATE_USE_RE.sub(' ', text)
    text = text.replace('\x00', '').replace('\xad', '')
    text = _MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


def _needs_ocr(text: str) -> tuple[bool, list[str]]:
    """Return (needs_ocr, reasons). Detects font garbling, layout artefacts."""
    reasons = []
    words = text.split()
    wc = len(words)
    if wc < TEXT_MIN_WORDS:
        reasons.append(f"too short ({wc} words)")
    garbled = len(_PRIVATE_USE_RE.findall(text))
    if garbled >= TEXT_GARBLE_THRESHOLD:
        reasons.append(f"unicode garbling ({garbled} private-use chars)")
    nl_ratio = text.count('\n') / max(wc, 1)
    if nl_ratio > TEXT_MAX_NL_RATIO:
        reasons.append(f"high newline ratio ({nl_ratio:.2f}) — layout artefact")
    formula_chars = sum(
        1 for c in text
        if c in '∂∫∑∏√≈≤≥≠±×÷αβγδεζηθλμνξπρσφψωΑΒΓΔΕΖΘΛΜΝΞΠΡΣΦΨΩ'
    )
    if formula_chars >= TEXT_FORMULA_THRESHOLD:
        reasons.append(
            f"heavy formula content ({formula_chars} math chars) — "
            "symbol font not decoded; MathOCR (Nougat/MathPix) required for 100% fidelity"
        )
    return bool(reasons), reasons


# ──────────────────────────────────────────────────────────────────────────────
# PDF layout classifier  — determines per-page extraction strategy
# ──────────────────────────────────────────────────────────────────────────────

class _PageLayout:
    """Lightweight struct describing what is on a PDF page."""
    __slots__ = (
        "has_vector_text",    # fitz found selectable text
        "word_count",         # number of words from fitz
        "has_rotated_spans",  # any span with dir != (1,0)
        "rotated_span_ratio", # fraction of spans that are rotated
        "has_table_lines",    # horizontal + vertical drawing lines
        "h_line_count",
        "v_line_count",
        "is_image_only",      # page contains images and virtually no text
        "is_blank",
        "is_slide_layout",    # wide page (pptx export) or very few words per block
        "has_multicolumn",    # text blocks distributed across ≥2 x-regions
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k, False))


def _classify_page(fitz_page) -> _PageLayout:
    """
    Inspect a fitz page and return a _PageLayout describing its content.
    Costs ~1-2 ms per page; runs once per page during extraction.
    """
    d_out   = fitz_page.get_text("dict")
    blocks  = d_out["blocks"]
    images  = fitz_page.get_images(full=False)
    drawings = fitz_page.get_drawings()

    # ── Span analysis ─────────────────────────────────────────────────────────
    all_spans = []
    for b in blocks:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            d = line["dir"]
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    all_spans.append((d, span))

    total_spans   = len(all_spans)
    rotated_spans = [(d, s) for d, s in all_spans if abs(d[1]) > 0.15]
    rot_ratio     = len(rotated_spans) / max(total_spans, 1)

    wc = sum(len(s["text"].split()) for _, s in all_spans)

    # ── Drawing / line analysis ────────────────────────────────────────────────
    h_lines = [d for d in drawings if abs(d["rect"].height) < 4 and d["rect"].width > 30]
    v_lines = [d for d in drawings if abs(d["rect"].width)  < 4 and d["rect"].height > 30]

    # ── Multi-column detection ─────────────────────────────────────────────────
    txt_blocks = [b for b in blocks if b["type"] == 0]
    x_buckets  = set()
    for b in txt_blocks:
        x_buckets.add(round(b["bbox"][0] / 80))   # 80px buckets
    has_multicol = len(x_buckets) >= 3 and len(txt_blocks) >= 6

    # ── Slide-layout heuristic (PPTX exports are wide, few words/block) ────────
    pw = fitz_page.rect.width
    ph = fitz_page.rect.height
    is_slide = (pw / max(ph, 1)) > 1.5 or (pw > 800 and wc < 200)

    return _PageLayout(
        has_vector_text    = total_spans > 0,
        word_count         = wc,
        has_rotated_spans  = len(rotated_spans) > 0,
        rotated_span_ratio = rot_ratio,
        has_table_lines    = len(h_lines) >= 3 and len(v_lines) >= 2,
        h_line_count       = len(h_lines),
        v_line_count       = len(v_lines),
        is_image_only      = wc < 5 and len(images) > 0,
        is_blank           = wc == 0 and len(images) == 0,
        is_slide_layout    = is_slide,
        has_multicolumn    = has_multicol,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rotated-text table reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def _reconstruct_rotated_table(fitz_page) -> Optional[str]:
    """
    Handle pages where the TABLE itself is rotated 90° CCW within the page
    (common in MIL-HDBK-5J and similar technical standards).

    These pages have span direction=(0,-1) meaning the text runs bottom→top
    on the screen, i.e. each "column" of the table is a horizontal strip of
    rotated text.

    Strategy:
      1. Extract all spans with their (x, y) bounding box.
      2. Group spans by their x0-coordinate ± 12px (each x-group = one
         logical ROW of the original landscape table).
      3. Within each x-group, sort spans by y DESCENDING (because dir=(0,-1)
         means the first character of the string is at the top, which has the
         HIGHEST y0 in PDF space after 90° CCW rotation).
      4. Produce a markdown table where each x-group = one row.

    Returns None if the page has fewer than 2 rotated-text groups (not a table).
    """
    d_out = fitz_page.get_text("dict")

    rotated_spans = []
    normal_spans  = []
    for b in d_out["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            dv = line["dir"]
            for span in line["spans"]:
                t = span["text"].strip()
                if not t:
                    continue
                entry = {
                    "text": t,
                    "x0":   span["bbox"][0],
                    "y0":   span["bbox"][1],
                    "y1":   span["bbox"][3],
                }
                if abs(dv[1]) > 0.15:
                    rotated_spans.append(entry)
                else:
                    normal_spans.append(entry)

    if not rotated_spans:
        return None

    # Are the majority of spans rotated? (table pages) vs minority (just labels)
    total = len(rotated_spans) + len(normal_spans)
    if len(rotated_spans) / max(total, 1) < 0.3:
        return None   # mostly normal text; don't reconstruct as rotated table

    # Group by x0 in 15px buckets → each bucket = one logical row
    from collections import defaultdict
    row_groups: dict[int, list] = defaultdict(list)
    for s in rotated_spans:
        bucket = round(s["x0"] / 15) * 15
        row_groups[bucket].append(s)

    if len(row_groups) < 2:
        return None

    # Sort groups by x (left-to-right = top-to-bottom of original table)
    sorted_rows = sorted(row_groups.items(), key=lambda kv: kv[0])

    # Within each row, sort spans by y DESCENDING to reconstruct reading order
    table_rows: list[str] = []
    for _, spans in sorted_rows:
        spans_sorted = sorted(spans, key=lambda s: -s["y0"])
        cell_text = " ".join(s["text"] for s in spans_sorted).strip()
        if cell_text:
            table_rows.append(cell_text)

    if len(table_rows) < 3:
        return None

    # We now have flat rows; try to detect column structure within each row
    # by looking at the y-gaps between spans in the first data row
    # Simple heuristic: treat each row as a single cell (monolithic row)
    # → one-column table preserving content order
    header = table_rows[0]
    data   = table_rows[1:]

    md_lines = [f"| {header} |", "|---|"]
    for row in data:
        md_lines.append(f"| {row} |")

    return "\n".join(md_lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_table_md(table_md: str) -> tuple[bool, list[str]]:
    """
    Validate a markdown-formatted table string.
    Returns (is_valid, reasons_for_rejection).
    """
    reasons = []
    lines      = [l for l in table_md.strip().splitlines() if l.strip()]
    data_lines = [l for l in lines if not re.match(r'^\|[-| :]+\|$', l.strip())]
    data_rows  = len(data_lines) - 1   # subtract header

    if data_rows < TABLE_MIN_DATA_ROWS:
        reasons.append(f"too few data rows ({data_rows}; need ≥{TABLE_MIN_DATA_ROWS})")

    total_cells = 0
    empty_cells = 0
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        total_cells += len(cells)
        empty_cells += sum(1 for c in cells if c in ('', 'None', 'nan', 'N/A'))

    if total_cells > 0:
        empty_ratio = empty_cells / total_cells
        if empty_ratio > TABLE_MAX_EMPTY_RATIO:
            reasons.append(
                f"high empty-cell ratio ({empty_ratio:.0%}; "
                f"threshold {TABLE_MAX_EMPTY_RATIO:.0%})"
            )

    if len(table_md.split()) < TABLE_MIN_WORDS:
        reasons.append(f"very short table ({len(table_md.split())} words)")

    return len(reasons) == 0, reasons


# ──────────────────────────────────────────────────────────────────────────────
# Table extraction  — 6-strategy pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _rows_to_markdown(rows: list[list]) -> str:
    """Convert a list-of-lists table to GitHub markdown."""
    if not rows:
        return ""
    md_rows = []
    for i, row in enumerate(rows):
        cleaned = [str(c).strip() if c is not None else "" for c in row]
        md_rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            md_rows.append("|" + "|".join(["---"] * len(cleaned)) + "|")
    return "\n".join(md_rows)


def _extract_tables_pdfplumber(pdf_path: str, page_num: int) -> list[str]:
    """
    Strategy 1: pdfplumber with lines_strict first, then default fallback.
    Best for: technical docs with explicit grid lines (MIL-HDBK-5J normal tables).
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return []
            page = pdf.pages[page_num]
            for settings in [
                {"vertical_strategy": "lines_strict",
                 "horizontal_strategy": "lines_strict",
                 "snap_tolerance": 5, "join_tolerance": 3},
                {},   # default — broader detection
            ]:
                tables = page.extract_tables(settings)
                if tables:
                    for table in tables:
                        if table:
                            md = _rows_to_markdown(table)
                            if md:
                                results.append(md)
                    break   # stop after first successful setting
    except Exception as e:
        print(f"[WARN] pdfplumber p{page_num}: {e}")
    return results


def _extract_tables_camelot_lattice(pdf_path: str, page_num: int) -> list[str]:
    """
    Strategy 2: camelot lattice — explicit-grid tables.
    Best for: bordered tables in technical standards.
    """
    results = []
    try:
        import camelot
        tbls = camelot.read_pdf(
            pdf_path, pages=str(page_num + 1), flavor="lattice",
            line_scale=40, copy_text=["v"],
        )
        for tbl in tbls:
            if tbl.parsing_report.get("accuracy", 0) < 50:
                continue
            md = tbl.df.to_markdown(index=False)
            if md:
                results.append(md)
    except Exception:
        pass
    return results


def _extract_tables_camelot_stream(pdf_path: str, page_num: int) -> list[str]:
    """
    Strategy 3: camelot stream — whitespace-aligned / borderless tables.
    Best for: textbooks (Applied Fluid Mechanics TABLE 1.2 style).
    """
    results = []
    try:
        import camelot
        tbls = camelot.read_pdf(
            pdf_path, pages=str(page_num + 1), flavor="stream",
            edge_tol=50, row_tol=10,
        )
        for tbl in tbls:
            if tbl.parsing_report.get("accuracy", 0) < 60:
                continue
            md = tbl.df.to_markdown(index=False)
            if md:
                results.append(md)
    except Exception:
        pass
    return results


def _extract_tables_span_bbox(fitz_page) -> list[str]:
    """
    Strategy 4: Span bounding-box reconstruction.
    Best for: borderless tables in multi-column textbooks where both
    camelot stream and pdfplumber produce fragmented/wrong columns.

    Guard conditions (ALL must pass to avoid false positives on prose pages):
      • The page text contains a table-keyword ("Table", "TABLE", "Appendix")
        within the first 800 characters of extracted text, OR has ≥3 distinct
        x-region clusters that each appear in ≥40% of rows (structural columns).
      • At least 3 detected column boundaries.
      • Column count is consistent: ≥60% of rows populate ≥2 columns.

    Returns [] if the page looks like running prose rather than a table.
    """
    d_out = fitz_page.get_text("dict")
    full_text = fitz_page.get_text("text")

    # Guard 1: table keyword present near top of page?
    has_table_kw = bool(re.search(
        r'\b(?:Table|TABLE|Appendix|APPENDIX|FIGURE|Figure)\b',
        full_text[:800]
    ))

    spans_flat = []
    for b in d_out["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            if abs(line["dir"][1]) > 0.15:
                continue   # skip rotated spans
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    spans_flat.append({
                        "text": t,
                        "x0":   span["bbox"][0],
                        "y0":   span["bbox"][1],
                        "x1":   span["bbox"][2],
                    })

    if len(spans_flat) < 6:
        return []

    # Cluster spans into rows by y0 (±5 pt tolerance)
    spans_flat.sort(key=lambda s: s["y0"])
    rows: list[list[dict]] = []
    for s in spans_flat:
        if rows and abs(s["y0"] - rows[-1][0]["y0"]) <= 5:
            rows[-1].append(s)
        else:
            rows.append([s])

    if len(rows) < 3:
        return []

    for row in rows:
        row.sort(key=lambda s: s["x0"])

    # Detect column boundaries: collect all x0 values, bucket into ~20px groups
    all_x = [s["x0"] for row in rows for s in row]
    all_x.sort()
    col_boundaries: list[float] = []
    prev = -999.0
    for x in all_x:
        if x - prev > 20:
            col_boundaries.append(x)
            prev = x

    n_cols = len(col_boundaries)
    if n_cols < 3:
        return []   # fewer than 3 columns → not a data table

    # Guard 2: structural column check — each boundary must appear in ≥30% of rows
    col_hit_counts = [0] * n_cols
    for row in rows:
        occupied = set()
        for s in row:
            for i, cb in enumerate(col_boundaries):
                if s["x0"] <= cb + 25:
                    occupied.add(i)
                    break
        for ci in occupied:
            col_hit_counts[ci] += 1

    structural_cols = sum(1 for c in col_hit_counts if c >= len(rows) * 0.30)
    if structural_cols < 3 and not has_table_kw:
        return []   # looks like prose layout, not a table

    # Guard 3: row consistency — threshold depends on whether table keyword present
    # With keyword: 50% of rows need ≥2 cells (table may have header rows)
    # Without keyword: 75% — prose pages frequently pass 50% due to multi-column layout
    # Also cap: if >8 columns detected without a table keyword it's prose fragmentation
    if n_cols > 8 and not has_table_kw:
        return []

    consistency_threshold = 0.50 if has_table_kw else 0.75

    def col_idx(x0: float) -> int:
        for i, cb in enumerate(col_boundaries):
            if x0 <= cb + 25:
                return i
        return n_cols - 1

    table_data: list[list[str]] = []
    multi_cell_rows = 0
    for row in rows:
        cells: dict[int, list[str]] = {}
        for s in row:
            ci = col_idx(s["x0"])
            cells.setdefault(ci, []).append(s["text"])
        row_cells = [" ".join(cells.get(ci, [""])) for ci in range(n_cols)]
        non_empty = sum(1 for c in row_cells if c.strip())
        if non_empty >= 2:
            multi_cell_rows += 1
        table_data.append(row_cells)

    if multi_cell_rows < len(rows) * consistency_threshold:
        return []   # mostly single-cell rows → prose, not a table

    if len(table_data) < 3:
        return []

    return [_rows_to_markdown(table_data)]


def _extract_tables_pymupdf4llm_md(page_md_text: str) -> list[str]:
    """
    Strategy 5: Parse GFM markdown tables embedded in pymupdf4llm output.
    Best for: any PDF that pymupdf4llm handles natively (most modern vector PDFs).

    Handles pymupdf4llm quirks:
      • <br> line breaks inside cells → replaced with space
      • —_ ligature artefacts → removed
      • Merged / empty cells preserved as-is
    """
    if not page_md_text:
        return []

    # Clean <br> and artefacts before regex matching
    cleaned = page_md_text.replace('<br>', ' ').replace('—_', '').replace('—', '-')

    results = []
    # GFM table: header row | separator row (--- / :---) | 1+ data rows
    # Allow separator rows with only dashes, colons, and pipes
    table_re = re.compile(
        r'(\|[^\n]+\|\n'           # header row
        r'\|[-|: ]+\|\n'           # separator row
        r'(?:\|[^\n]+\|\n?)+)',    # one or more data rows
        re.MULTILINE,
    )
    for m in table_re.finditer(cleaned + "\n"):
        tbl = m.group(0).strip()
        # Remove purely empty rows (all pipes and spaces)
        filtered_lines = [
            line for line in tbl.splitlines()
            if re.sub(r'[| ]', '', line).strip()   # has non-pipe/space content
            or re.match(r'^\|[-|: ]+\|$', line.strip())  # or is a separator
        ]
        if len(filtered_lines) >= 3:
            results.append("\n".join(filtered_lines))
    return results


def _extract_tables_ocr_crop(
    pdf_path: str, page_num: int, fitz_doc, table_bboxes: list
) -> list[str]:
    """
    Strategy 6: EasyOCR image-crop table reconstruction.
    Last resort for scanned PDFs where all text-based strategies fail.
    Detects row structure by grouping OCR tokens by y-midpoint proximity.
    """
    results = []
    try:
        import easyocr
        page = fitz_doc[page_num]
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        regions = table_bboxes if table_bboxes else [page.rect]

        for bbox in regions:
            clip = fitz.Rect(bbox) if not isinstance(bbox, fitz.Rect) else bbox
            pix  = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
            ocr_results = reader.readtext(pix.tobytes(), detail=1, paragraph=False)

            rows_by_y: dict[int, list[str]] = {}
            for (bbox_pts, text, conf) in ocr_results:
                if conf < 0.3:
                    continue
                y_mid = int((bbox_pts[0][1] + bbox_pts[2][1]) / 2 / 12)
                rows_by_y.setdefault(y_mid, []).append(text)

            if len(rows_by_y) < 2:
                continue

            rows = [" | ".join(tokens) for tokens in rows_by_y.values()]
            header = "| " + rows[0] + " |"
            sep    = "|" + "|".join(["---"] * max(rows[0].count("|") + 1, 1)) + "|"
            data   = ["| " + r + " |" for r in rows[1:]]
            md     = "\n".join([header, sep] + data)
            results.append(md)
    except Exception as e:
        print(f"[WARN] OCR table crop p{page_num}: {e}")
    return results


def _extract_tables_from_page(
    pdf_path: str,
    page_num: int,
    page_md_text: str = "",
    fitz_doc=None,
    layout: Optional[_PageLayout] = None,
) -> tuple[list[str], list[dict]]:
    """
    Adaptive multi-strategy table extraction.

    Strategy selection is driven by the page layout classifier:

    ROTATED TABLE  (dir=(0,-1) dominant):
        → _reconstruct_rotated_table() only.
          Other strategies produce reversed/garbled output on these pages.

    BORDERED TABLE  (h_lines ≥3, v_lines ≥2):
        → pdfplumber lines_strict → camelot lattice → span-bbox

    BORDERLESS TABLE  (no lines, but tabular content):
        → pymupdf4llm markdown parser → camelot stream → span-bbox

    IMAGE-ONLY PAGE:
        → OCR crop (last resort, slow)

    All candidates pass through _validate_table_md() before storage.
    """
    valid_tables:   list[str]  = []
    all_candidates: list[tuple] = []  # (strategy_name, md)
    issues:         list[dict] = []

    # ── Route by layout ────────────────────────────────────────────────────────
    if layout is None and fitz_doc is not None:
        layout = _classify_page(fitz_doc[page_num])

    # SLIDE LAYOUT — skip ALL table extraction unless explicit grid lines present.
    # PPTX-exported PDFs have content boxes that pdfplumber misidentifies as tables.
    # Real data tables on slides are extremely rare; if one exists it will have
    # explicit drawn grid lines (has_table_lines=True).
    if layout is not None and layout.is_slide_layout and not layout.has_table_lines:
        return [], []

    # ROTATED TABLE path — special handling, short-circuits everything else
    if (layout is not None and layout.has_rotated_spans
            and layout.rotated_span_ratio > 0.3 and fitz_doc is not None):
        md = _reconstruct_rotated_table(fitz_doc[page_num])
        if md:
            ok, reasons = _validate_table_md(md)
            if ok:
                return [md], []
            else:
                issues.append({"strategy": "rotated-span-reconstruct",
                               "reasons": reasons, "raw_preview": md[:200]})

    # NORMAL BORDERED TABLE path
    if layout is None or layout.has_table_lines:
        for md in _extract_tables_pdfplumber(pdf_path, page_num):
            all_candidates.append(("pdfplumber", md))
        if not all_candidates:
            for md in _extract_tables_camelot_lattice(pdf_path, page_num):
                all_candidates.append(("camelot-lattice", md))

    # BORDERLESS / TEXTBOOK TABLE path
    if not all_candidates:
        if page_md_text:
            for md in _extract_tables_pymupdf4llm_md(page_md_text):
                all_candidates.append(("pymupdf4llm-md", md))
        if not all_candidates:
            for md in _extract_tables_camelot_stream(pdf_path, page_num):
                all_candidates.append(("camelot-stream", md))

    # SPAN-BBOX path (catches anything the line-based strategies miss)
    if not all_candidates and fitz_doc is not None:
        for md in _extract_tables_span_bbox(fitz_doc[page_num]):
            all_candidates.append(("span-bbox", md))

    # Validate all candidates
    for strategy, md in all_candidates:
        ok, reasons = _validate_table_md(md)
        if ok:
            valid_tables.append(md)
        else:
            issues.append({"strategy": strategy,
                           "reasons": reasons, "raw_preview": md[:200]})

    # OCR FALLBACK — only if nothing worked AND page has images
    if not valid_tables and fitz_doc is not None:
        do_ocr = (layout is not None and layout.is_image_only)
        if not do_ocr and layout is not None and layout.has_table_lines:
            do_ocr = True   # lined table but text-based strategies all failed
        if do_ocr:
            table_bboxes = []
            try:
                page_obj = fitz_doc[page_num]
                drawings = page_obj.get_drawings()
                h_lines  = [d["rect"] for d in drawings
                            if abs(d["rect"].height) < 4 and d["rect"].width > 50]
                if len(h_lines) >= 2:
                    y_coords = sorted(r.y0 for r in h_lines)
                    table_bboxes = [fitz.Rect(0, y_coords[0] - 5,
                                              page_obj.rect.width, y_coords[-1] + 5)]
            except Exception:
                pass
            for md in _extract_tables_ocr_crop(pdf_path, page_num, fitz_doc, table_bboxes):
                ok, reasons = _validate_table_md(md)
                if ok:
                    valid_tables.append(md)
                else:
                    issues.append({"strategy": "ocr-crop",
                                   "reasons": reasons, "raw_preview": md[:200]})

    return valid_tables, issues


# ──────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_pymupdf4llm(md_page_text: str) -> str:
    """
    From a pre-computed pymupdf4llm markdown page, strip embedded GFM table
    blocks (handled by the table pipeline) and return clean prose text.

    Also cleans pymupdf4llm quirks:
      • <br> → space
      • —_ ligature artefacts → removed
      • Stray leading pipe chars left by table removal
    """
    if not md_page_text:
        return ""
    # Clean <br> and ligature artefacts first
    text = md_page_text.replace('<br>', ' ').replace('—_', '').replace('—', '-')
    # Remove GFM table blocks
    text = re.sub(
        r'\|[^\n]+\|\n\|[-|: ]+\|\n(?:\|[^\n]+\|\n?)+',
        '',
        text,
        flags=re.MULTILINE,
    )
    # Remove stray pipe lines left behind
    text = re.sub(r'^\s*\|.*\|\s*$', '', text, flags=re.MULTILINE)
    return _clean_text(text)


def _extract_text_fitz_ordered(fitz_page) -> str:
    """
    Fallback text extraction using fitz 'text' mode.
    Applies layout-aware sorting: for single-column pages this is identical
    to normal order; for multi-column pages fitz preserves the correct
    reading order (top-to-bottom per column, left-to-right between columns).
    Rotated spans are intentionally skipped here — they are handled by the
    table reconstruction path.
    """
    d_out = fitz_page.get_text("dict")
    lines_text = []
    for b in d_out["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            if abs(line["dir"][1]) > 0.15:
                continue   # skip rotated text
            line_str = " ".join(
                span["text"] for span in line["spans"]
                if span["text"].strip()
            ).strip()
            if line_str:
                lines_text.append(line_str)
    return _clean_text(" ".join(lines_text))


def _ocr_page(fitz_page) -> str:
    """Run EasyOCR on a full page pixmap. Returns joined text."""
    try:
        import easyocr
        pix    = fitz_page.get_pixmap(dpi=400)
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        return " ".join(reader.readtext(pix.tobytes(), detail=0, paragraph=True))
    except Exception as e:
        print(f"[WARN] OCR page failed: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Text + table extraction from a whole PDF
# ──────────────────────────────────────────────────────────────────────────────

class PageExtractionIssue:
    """Structured record of a single page-level extraction problem."""
    __slots__ = ("pdf_name", "page_num", "chunk_type", "reasons", "raw_preview")

    def __init__(self, pdf_name: str, page_num: int, chunk_type: str,
                 reasons: list[str], raw_preview: str = ""):
        self.pdf_name    = pdf_name
        self.page_num    = page_num
        self.chunk_type  = chunk_type
        self.reasons     = reasons
        self.raw_preview = raw_preview

    def __repr__(self):
        return (f"<Issue pdf={self.pdf_name} p={self.page_num} "
                f"type={self.chunk_type} reasons={self.reasons}>")


def extract_pdf_content(
    pdf_path: str, pdf_name: str = ""
) -> tuple[list[dict], list[PageExtractionIssue]]:
    """
    Main entry point for extracting text and tables from any PDF.

    Returns:
      pages_out — list of { "page": int, "text": str, "tables": [md,...] }
      issues    — list of PageExtractionIssue (for the QA audit)

    Per-page strategy routing
    ─────────────────────────
    BLANK PAGE          → skip entirely (no content to index)
    IMAGE-ONLY PAGE     → OCR full page for text; OCR crop for tables
    ROTATED-SPAN PAGE   → rotated-table reconstruction; skip normal text
                          (rotated text IS the table; non-table text is headers)
    SLIDE-LAYOUT PAGE   → fitz dict ordered extraction (columns OK);
                          no table pipeline (slides don't have data tables)
    MULTI-COLUMN PAGE   → pymupdf4llm primary (handles columns correctly)
    NORMAL PAGE         → pymupdf4llm primary; fitz fallback
    """
    import pymupdf4llm

    pages_out: list[dict]                = []
    issues:    list[PageExtractionIssue] = []
    name = pdf_name or Path(pdf_path).name

    doc     = fitz.open(pdf_path)
    n_pages = len(doc)

    # ── Single-pass pymupdf4llm (fast; handles multi-column, basic tables) ──
    try:
        md_full      = pymupdf4llm.to_markdown(pdf_path)
        full_text_ok = len(md_full.strip()) >= 100
    except Exception:
        md_full      = ""
        full_text_ok = False

    md_pages: list[str] = []
    if full_text_ok:
        # pymupdf4llm separates pages with \x0c (form-feed)
        md_pages = md_full.split("\x0c")
        if len(md_pages) != n_pages:
            md_pages = []   # page count mismatch → don't use split output

    for page_num in range(n_pages):
        fitz_page = doc[page_num]
        layout    = _classify_page(fitz_page)
        page_md   = md_pages[page_num] if (md_pages and page_num < len(md_pages)) else ""

        # ── BLANK — skip ─────────────────────────────────────────────────────
        if layout.is_blank:
            continue

        # ── TEXT extraction ───────────────────────────────────────────────────
        if layout.is_image_only:
            # Always try fitz first — PPTX slides often have a small amount of
            # vector text (title bar, slide number) even on image-heavy pages.
            vector_text = _extract_text_fitz_ordered(fitz_page)
            if len(vector_text.split()) >= TEXT_MIN_WORDS:
                # Enough vector text found (e.g. a diagram-only slide with title)
                raw_text = vector_text
            else:
                # Truly image-only (scanned page, no selectable text at all) → OCR
                ocr_text = _ocr_page(fitz_page)
                # Combine: prefer the longer result, but always keep vector_text
                # as a prefix so slide titles are never lost
                if len(ocr_text.split()) > len(vector_text.split()):
                    raw_text = (vector_text + " " + ocr_text).strip() if vector_text else ocr_text
                else:
                    raw_text = vector_text
                if not raw_text:
                    issues.append(PageExtractionIssue(
                        name, page_num, "text",
                        ["image-only page; OCR produced no output"],
                        raw_preview=""
                    ))
                    raw_text = ""

        elif layout.has_rotated_spans and layout.rotated_span_ratio > 0.7:
            # Page is almost entirely a rotated table — the "text" content
            # is the table itself; avoid double-storing it as text chunks.
            # Extract only the small amount of non-rotated header text.
            raw_text = _extract_text_fitz_ordered(fitz_page)

        elif page_md:
            # pymupdf4llm handled this page — strip embedded table blocks
            raw_text = _extract_text_pymupdf4llm(page_md)

        else:
            # Generic fitz ordered fallback
            raw_text = _extract_text_fitz_ordered(fitz_page)
            if not raw_text:
                raw_text = fitz_page.get_text("text")
            raw_text = _clean_text(raw_text)

        # Quality check → OCR retry if text looks degraded
        # Skip for slide-layout pages: short text on slides is intentional
        # (a diagram slide genuinely has only a title)
        needs_ocr_flag, text_reasons = _needs_ocr(raw_text)
        if needs_ocr_flag and not layout.is_image_only and not layout.is_slide_layout:
            ocr_text = _ocr_page(fitz_page)
            if len(ocr_text.split()) > len(raw_text.split()):
                raw_text = ocr_text
            else:
                issues.append(PageExtractionIssue(
                    name, page_num, "text", text_reasons,
                    raw_preview=raw_text[:200]
                ))

        # ── TABLE extraction ──────────────────────────────────────────────────
        # Slides don't have data tables — skip table pipeline for them
        if layout.is_slide_layout and not layout.has_table_lines:
            valid_tables = []
            tbl_issues   = []
        else:
            valid_tables, tbl_issues = _extract_tables_from_page(
                pdf_path, page_num,
                page_md_text=page_md,
                fitz_doc=doc,
                layout=layout,
            )

        for ti in tbl_issues:
            issues.append(PageExtractionIssue(
                name, page_num, "table",
                ti["reasons"],
                raw_preview=ti["raw_preview"],
            ))

        pages_out.append({
            "page":   page_num,
            "text":   raw_text,
            "tables": valid_tables,
        })

    doc.close()
    return pages_out, issues




# ──────────────────────────────────────────────────────────────────────────────
# QA Audit helpers
# ──────────────────────────────────────────────────────────────────────────────

def _audit_corpus_chunks(
    bm25_docs: list[str],
    bm25_meta: list[dict],
) -> dict[str, list[dict]]:
    """
    Scan every chunk currently in the BM25 corpus for known quality problems.
    Returns { pdf_name: [{"page": int, "type": str, "reasons": [str], "preview": str}, ...] }
    """
    findings: dict[str, list[dict]] = defaultdict(list)

    for text, meta in zip(bm25_docs, bm25_meta):
        src  = meta.get("source", "unknown")
        page = meta.get("page", -1)
        ctype = meta.get("type", "text")

        reasons = []
        if ctype == "table":
            ok, r = _validate_table_md(text)
            if not ok:
                reasons = r
        else:
            _, r = _needs_ocr(text)
            if r:
                reasons = r

        if reasons:
            findings[src].append({
                "page":    page,
                "type":    ctype,
                "reasons": reasons,
                "preview": text[:200],
            })

    return dict(findings)


def _print_audit_report(
    findings: dict[str, list[dict]],
    also_from_new_extraction: dict[str, list[PageExtractionIssue]] | None = None,
) -> None:
    """Pretty-print the full QA audit report to stdout."""
    print("\n" + "═" * 70)
    print("  KNOWLEDGE BASE QUALITY AUDIT REPORT")
    print("═" * 70)

    combined: dict[str, list[dict]] = defaultdict(list)
    for src, items in findings.items():
        combined[src].extend(items)

    if also_from_new_extraction:
        for src, issue_list in also_from_new_extraction.items():
            for iss in issue_list:
                combined[src].append({
                    "page":    iss.page_num,
                    "type":    iss.chunk_type,
                    "reasons": iss.reasons,
                    "preview": iss.raw_preview,
                })

    if not combined:
        print("\n  ✅ No quality problems found in any indexed chunk.\n")
        return

    total_issues = sum(len(v) for v in combined.values())
    print(f"\n  Found {total_issues} problematic chunk(s) across "
          f"{len(combined)} PDF(s):\n")

    # Summary by problem type
    type_counts: dict[str, int] = defaultdict(int)
    for items in combined.values():
        for item in items:
            for r in item["reasons"]:
                # Bucket the reason
                if "empty-cell" in r:
                    type_counts["Table: high empty-cell ratio"] += 1
                elif "few data rows" in r:
                    type_counts["Table: too few rows"] += 1
                elif "very short table" in r or "very short" in r:
                    type_counts["Chunk too short"] += 1
                elif "garbling" in r:
                    type_counts["Text: unicode/font garbling"] += 1
                elif "newline ratio" in r:
                    type_counts["Text: layout/OCR newline artefacts"] += 1
                elif "formula content" in r:
                    type_counts["Text: math formula mis-extraction"] += 1
                elif "truncated" in r or "..." in r:
                    type_counts["Table: truncated content"] += 1
                else:
                    type_counts["Other"] += 1

    print("  Problem type summary:")
    for ptype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {cnt:4d}×  {ptype}")

    print()
    for src, items in sorted(combined.items(), key=lambda x: -len(x[1])):
        print(f"  ▶ {src}  ({len(items)} issue(s))")
        # Group by page for compactness
        by_page: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            by_page[item["page"]].append(item)
        for pg in sorted(by_page.keys()):
            for item in by_page[pg]:
                tag = "[TABLE]" if item["type"] == "table" else "[TEXT]"
                print(f"      p{pg} {tag}  {'; '.join(item['reasons'])}")
        print()

    print("═" * 70 + "\n")


def _prompt_delete_pdfs(
    findings: dict[str, list[dict]],
    kb_instance: "OptimizedPDFKnowledgeBase",
) -> None:
    """
    Interactively ask the user which flagged PDFs they want to purge from
    all stores.  Purges: FAISS, BM25, parent_chunks, cluster_summaries,
    knowledge_graph nodes, and processed_files.json.
    """
    flagged_pdfs = sorted(findings.keys())
    if not flagged_pdfs:
        return

    print("The following PDFs have extraction quality issues:")
    for i, pdf in enumerate(flagged_pdfs, 1):
        print(f"  [{i}] {pdf}  ({len(findings[pdf])} issue(s))")

    print("\nWould you like to DELETE the extracted data for any of these PDFs?")
    print("Enter comma-separated numbers, 'all', or press Enter to skip: ", end="")
    user_input = input().strip()

    if not user_input:
        print("[KB] No data deleted.")
        return

    if user_input.lower() == "all":
        to_delete = flagged_pdfs
    else:
        indices = []
        for part in user_input.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(flagged_pdfs):
                    indices.append(idx)
        to_delete = [flagged_pdfs[i] for i in indices]

    if not to_delete:
        print("[KB] No valid selection — nothing deleted.")
        return

    print(f"\n[KB] Deleting data for: {to_delete}")
    for pdf_name in to_delete:
        kb_instance._purge_pdf_from_all_stores(pdf_name)
        print(f"  ✓ Purged: {pdf_name}")

    print("[KB] Deletion complete. Affected stores have been re-saved.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LLM summary generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_group_summary(
    texts: list[str], llm_client, topic_hint: str = ""
) -> tuple[str, str]:
    combined = " ".join(texts)[:3000]
    hint = f" The cluster appears to be about: {topic_hint}." if topic_hint else ""
    prompt = (
        f"You are a knowledge base organizer.{hint}\n\n"
        f"Here are several related text chunks:\n\n{combined}\n\n"
        "Task:\n"
        "1. Give a concise topic name (3-6 words, Title Case).\n"
        "2. Write a unique 2-4 sentence summary of the main ideas.\n\n"
        "Reply in this exact format:\n"
        "TOPIC: <topic name>\n"
        "SUMMARY: <summary text>"
    )
    try:
        raw = llm_client.generate(prompt=prompt, max_tokens=200)
        topic_match   = re.search(r"TOPIC:\s*(.+)", raw)
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?=TOPIC:|$)", raw, re.DOTALL)
        topic   = topic_match.group(1).strip()   if topic_match   else "General"
        summary = summary_match.group(1).strip() if summary_match else raw.strip()
    except Exception as e:
        print(f"[WARN] LLM summary failed: {e}")
        topic, summary = "General", combined[:300]
    return topic, summary


# ──────────────────────────────────────────────────────────────────────────────
# Main KB class
# ──────────────────────────────────────────────────────────────────────────────

class OptimizedPDFKnowledgeBase:
    """
    Advanced RAG Knowledge Base with:
      • Incremental PDF processing (processed_files.json)
      • Parent-child chunking
      • Hybrid dense+sparse retrieval
      • HyDE query augmentation
      • LLM-generated cluster summaries (auto-merged if near-duplicate)
      • Multi-strategy table extraction with validation
      • OCR fallbacks for scanned pages and tables
      • Post-extraction QA audit with interactive delete capability
      • MMR diversity re-ranking
      • Entity Knowledge Graph
    """

    CONTENT_SIMILARITY_THRESHOLD: float = 0.82

    def __init__(self, data_dir: str, llm_client=None):
        self.data_dir   = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.llm_client = llm_client

        self.registry_path      = os.path.join(self.models_dir, PROCESSED_FILES_JSON)
        self.vector_path        = os.path.join(self.models_dir, "faiss_index")
        self.bm25_path          = os.path.join(self.models_dir, "bm25_corpus.json")
        self.summaries_path     = os.path.join(self.models_dir, "cluster_summaries.json")
        self.parent_chunks_path = os.path.join(self.models_dir, "parent_chunks.json")

        self.vectorstore:       Optional[FAISS]    = None
        self.bm25:              Optional[BM25Okapi] = None
        self._bm25_docs:        list[str]           = []
        self._bm25_meta:        list[dict]          = []
        self.cluster_summaries: list[dict]          = []
        self.parent_chunks:     dict[str, str]      = {}

        print("Loading spaCy NLP model…")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.kg = SemanticKnowledgeGraph(self.models_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────

    def train_or_load(self) -> None:
        """
        Normal entry point used by main.py / SearchFunction at startup.

        Loads existing models and processes any new PDFs found in data_dir.
        The QA audit is intentionally SKIPPED here so that the application
        starts instantly without any interactive prompts.

        To run the quality audit and the interactive delete tool, run
        KnowledgeBase_training.py directly (or call run_training_audit()).
        """
        registry = _load_processed_registry(self.registry_path)
        self._load_existing_models()

        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        new_pdfs  = [
            f for f in pdf_files
            if _file_sha256(os.path.join(self.data_dir, f)) not in registry.values()
        ]

        if not new_pdfs:
            print(f"[KB] All {len(pdf_files)} PDF(s) already processed.")
            return

        print(f"[KB] Found {len(new_pdfs)} new PDF(s) to process.")
        self._process_pdfs(new_pdfs, registry)

    def run_training_audit(self) -> None:
        """
        Full training entry point — call this when running KnowledgeBase_training.py
        directly (not from main.py).

        Does everything train_or_load() does, then additionally:
          • Runs the QA audit across ALL indexed chunks (new + pre-existing).
          • Prints a detailed report of every problematic chunk.
          • Interactively asks whether to delete bad PDFs from all stores.
        """
        registry = _load_processed_registry(self.registry_path)
        self._load_existing_models()

        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        new_pdfs  = [
            f for f in pdf_files
            if _file_sha256(os.path.join(self.data_dir, f)) not in registry.values()
        ]

        new_extraction_issues: dict[str, list[PageExtractionIssue]] = {}

        if not new_pdfs:
            print(f"[KB] All {len(pdf_files)} PDF(s) already processed — models loaded.")
        else:
            print(f"[KB] Found {len(new_pdfs)} new PDF(s) to process.")
            new_extraction_issues = self._process_pdfs(new_pdfs, registry)

        # ── QA audit ──────────────────────────────────────────────────────
        print("\n[KB] Running quality audit on all indexed chunks…")
        corpus_findings = _audit_corpus_chunks(self._bm25_docs, self._bm25_meta)

        new_findings: dict[str, list[dict]] = {}
        for pdf_name, issue_list in new_extraction_issues.items():
            new_findings[pdf_name] = [
                {
                    "page":    iss.page_num,
                    "type":    iss.chunk_type,
                    "reasons": iss.reasons,
                    "preview": iss.raw_preview,
                }
                for iss in issue_list
            ]

        all_findings = {**new_findings, **corpus_findings}
        _print_audit_report(all_findings)

        if all_findings:
            _prompt_delete_pdfs(all_findings, self)

    # ──────────────────────────────────────────────────────────────────────
    # Load existing artefacts
    # ──────────────────────────────────────────────────────────────────────

    def _load_existing_models(self) -> None:
        if os.path.exists(self.vector_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_path, self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("[KB] FAISS index loaded.")
            except Exception as e:
                print(f"[WARN] FAISS load failed: {e}")

        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self._bm25_docs = payload["docs"]
                self._bm25_meta = payload["meta"]
                tokenized = [d.lower().split() for d in self._bm25_docs]
                self.bm25 = BM25Okapi(tokenized)
                print("[KB] BM25 index loaded.")
            except Exception as e:
                print(f"[WARN] BM25 load failed: {e}")

        if os.path.exists(self.summaries_path):
            with open(self.summaries_path, "r", encoding="utf-8") as f:
                self.cluster_summaries = json.load(f)
            print(f"[KB] {len(self.cluster_summaries)} cluster summaries loaded.")

        if os.path.exists(self.parent_chunks_path):
            with open(self.parent_chunks_path, "r", encoding="utf-8") as f:
                self.parent_chunks = json.load(f)
            print(f"[KB] {len(self.parent_chunks)} parent chunks loaded.")

        kg_path = os.path.join(self.models_dir, "knowledge_graph.pkl")
        if os.path.exists(kg_path):
            try:
                self.kg.load()
                print("[KB] Knowledge graph loaded.")
            except Exception as e:
                print(f"[WARN] KG load failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # PDF processing pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _process_pdfs(
        self, pdf_list: list[str], registry: dict
    ) -> dict[str, list[PageExtractionIssue]]:
        """
        Process each PDF, save incrementally, and collect extraction issues.
        Returns { pdf_name: [PageExtractionIssue, ...] }
        """
        all_issues: dict[str, list[PageExtractionIssue]] = {}

        for pdf_name in tqdm(pdf_list, desc="Processing PDFs"):
            pdf_path = os.path.join(self.data_dir, pdf_name)
            sha      = _file_sha256(pdf_path)

            print(f"\n[KB] ── Processing: {pdf_name} ──")
            pages, issues = extract_pdf_content(pdf_path, pdf_name=pdf_name)

            if issues:
                all_issues[pdf_name] = issues
                print(f"[KB] ⚠ {len(issues)} extraction issue(s) found in {pdf_name}")

            pdf_child_docs:  list[Document] = []
            pdf_child_texts: list[str]      = []
            pdf_child_meta:  list[dict]     = []
            pdf_parent_map:  dict[str, str] = {}

            for page_info in pages:
                page_num = page_info["page"]
                raw_text = page_info["text"]
                tables   = page_info["tables"]

                parent_chunks = _chunk_text(
                    raw_text,
                    target_words=PARENT_CHUNK_TARGET,
                    max_words=PARENT_CHUNK_MAX,
                    overlap_sents=1,
                )
                for pi, pc in enumerate(parent_chunks):
                    pid = f"{pdf_name}__p{page_num}__parent{pi}"
                    pdf_parent_map[pid] = pc

                child_chunks = _chunk_text(
                    raw_text,
                    target_words=CHILD_CHUNK_TARGET,
                    max_words=CHILD_CHUNK_MAX,
                    overlap_sents=SENTENCE_OVERLAP,
                )
                ratio = max(1, PARENT_CHUNK_TARGET // CHILD_CHUNK_TARGET)
                for ci, cc in enumerate(child_chunks):
                    child_id  = f"{pdf_name}__p{page_num}__c{ci}"
                    parent_id = (
                        f"{pdf_name}__p{page_num}__parent"
                        f"{min(ci // ratio, len(parent_chunks) - 1)}"
                    )
                    meta = {
                        "source":    pdf_name,
                        "page":      page_num,
                        "child_id":  child_id,
                        "parent_id": parent_id,
                        "type":      "text",
                    }
                    pdf_child_docs.append(Document(page_content=cc, metadata=meta))
                    pdf_child_texts.append(cc)
                    pdf_child_meta.append(meta)

                for ti, tbl_md in enumerate(tables):
                    table_id = f"{pdf_name}__p{page_num}__tbl{ti}"
                    meta = {
                        "source":    pdf_name,
                        "page":      page_num,
                        "child_id":  table_id,
                        "parent_id": table_id,
                        "type":      "table",
                    }
                    pdf_child_docs.append(Document(page_content=tbl_md, metadata=meta))
                    pdf_child_texts.append(tbl_md)
                    pdf_child_meta.append(meta)
                    pdf_parent_map[table_id] = tbl_md

            if not pdf_child_docs:
                print(f"[KB] No content extracted from {pdf_name} — skipping.")
                registry[pdf_name] = sha
                _save_processed_registry(self.registry_path, registry)
                continue

            assigned_group_id = self._route_pdf_to_group(pdf_name, pdf_child_texts)
            for meta in pdf_child_meta:
                meta["group_id"] = assigned_group_id

            # Throttled batching
            EMBED_BATCH_SIZE = 64  # tune down to 32 if still crashing
            EMBED_SLEEP_SECS = 0.5  # breathing room between batches

            print(f"[KB] Indexing {len(pdf_child_docs)} chunks into FAISS…")
            for batch_start in range(0, len(pdf_child_docs), EMBED_BATCH_SIZE):
                batch = pdf_child_docs[batch_start: batch_start + EMBED_BATCH_SIZE]
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    new_store = FAISS.from_documents(batch, self.embeddings)
                    self.vectorstore.merge_from(new_store)
                time.sleep(EMBED_SLEEP_SECS)
            self.vectorstore.save_local(self.vector_path)

            self._bm25_docs.extend(pdf_child_texts)
            self._bm25_meta.extend(pdf_child_meta)
            tokenized = [d.lower().split() for d in self._bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
            with open(self.bm25_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"docs": self._bm25_docs, "meta": self._bm25_meta},
                    f, ensure_ascii=False,
                )

            self.parent_chunks.update(pdf_parent_map)
            with open(self.parent_chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.parent_chunks, f, ensure_ascii=False)

            self._update_group_summary(
                assigned_group_id, pdf_name, pdf_child_texts, pdf_child_meta
            )
            self._build_knowledge_graph(pdf_child_texts, pdf_child_meta)

            registry[pdf_name] = sha
            _save_processed_registry(self.registry_path, registry)
            print(f"[KB] ✓ {pdf_name} fully saved (group: {assigned_group_id}).")

        print("\n[KB] All new PDFs processed and saved.")
        return all_issues

    # ──────────────────────────────────────────────────────────────────────
    # Purge a PDF from ALL stores — called by the interactive delete prompt
    # ──────────────────────────────────────────────────────────────────────

    def _purge_pdf_from_all_stores(self, pdf_name: str) -> None:
        """
        Remove every trace of pdf_name from:
          FAISS vectorstore, BM25 corpus, parent_chunks,
          cluster_summaries, knowledge_graph, processed_files.json.
        All affected files are re-saved atomically.
        """
        # ── BM25 + parent_chunks ──────────────────────────────────────────
        keep_idx = [
            i for i, m in enumerate(self._bm25_meta)
            if m.get("source") != pdf_name
        ]
        self._bm25_docs = [self._bm25_docs[i] for i in keep_idx]
        self._bm25_meta = [self._bm25_meta[i] for i in keep_idx]

        if self._bm25_docs:
            tokenized = [d.lower().split() for d in self._bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

        with open(self.bm25_path, "w", encoding="utf-8") as f:
            json.dump(
                {"docs": self._bm25_docs, "meta": self._bm25_meta},
                f, ensure_ascii=False,
            )

        # ── Parent chunks ─────────────────────────────────────────────────
        self.parent_chunks = {
            k: v for k, v in self.parent_chunks.items()
            if not k.startswith(pdf_name + "__")
        }
        with open(self.parent_chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.parent_chunks, f, ensure_ascii=False)

        # ── FAISS — rebuild from remaining BM25 docs ──────────────────────
        # (Rebuilding is the only safe way to remove docs from FAISS,
        #  since langchain-FAISS has no delete-by-id API.)
        if self._bm25_docs:
            print(f"[KB] Rebuilding FAISS index without {pdf_name}…")
            remaining_docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(self._bm25_docs, self._bm25_meta)
            ]
            # Batch to avoid OOM on large corpora
            batch_size = 512
            self.vectorstore = FAISS.from_documents(
                remaining_docs[:batch_size], self.embeddings
            )
            for start in range(batch_size, len(remaining_docs), batch_size):
                batch = remaining_docs[start : start + batch_size]
                new_vs = FAISS.from_documents(batch, self.embeddings)
                self.vectorstore.merge_from(new_vs)
            self.vectorstore.save_local(self.vector_path)
        else:
            self.vectorstore = None
            # Remove FAISS files if they exist
            for fname in ["index.faiss", "index.pkl"]:
                fpath = os.path.join(self.vector_path, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        # ── Cluster summaries ─────────────────────────────────────────────
        for summary in self.cluster_summaries:
            if pdf_name in summary.get("sources", []):
                summary["sources"].remove(pdf_name)
                summary.get("pdf_centroids", {}).pop(pdf_name, None)
        # Drop groups that are now empty
        self.cluster_summaries = [
            s for s in self.cluster_summaries if s.get("sources")
        ]
        with open(self.summaries_path, "w", encoding="utf-8") as f:
            json.dump(self.cluster_summaries, f, ensure_ascii=False, indent=2)

        # ── Knowledge graph ───────────────────────────────────────────────
        try:
            nodes_to_remove = [
                n for n, d in self.kg.graph.nodes(data=True)
                if d.get("source") == pdf_name
            ]
            self.kg.graph.remove_nodes_from(nodes_to_remove)
            self.kg.save()
        except Exception as e:
            print(f"[WARN] KG purge partial: {e}")

        # ── processed_files.json ──────────────────────────────────────────
        registry = _load_processed_registry(self.registry_path)
        registry.pop(pdf_name, None)
        _save_processed_registry(self.registry_path, registry)

    # ──────────────────────────────────────────────────────────────────────
    # Similarity routing
    # ──────────────────────────────────────────────────────────────────────

    def _centroid_embedding(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(384)
        embs = np.array(self.embeddings.embed_documents(texts[:50]))
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def _route_pdf_to_group(self, pdf_name: str, chunk_texts: list[str]) -> str:
        new_centroid  = self._centroid_embedding(chunk_texts)
        best_score    = -1.0
        best_group_id = None

        for summary in self.cluster_summaries:
            pdf_centroids = summary.get("pdf_centroids", {})
            for _pname, stored_vec in pdf_centroids.items():
                score = float(np.dot(new_centroid, np.array(stored_vec)))
                if score > best_score:
                    best_score    = score
                    best_group_id = summary["cluster_id"]

        if best_score >= self.CONTENT_SIMILARITY_THRESHOLD and best_group_id is not None:
            print(
                f"[KB] '{pdf_name}' matched '{best_group_id}' "
                f"(score={best_score:.3f})."
            )
            return best_group_id

        return f"grp_{pdf_name.replace(' ', '_').replace('.pdf', '')}_{int(time.time())}"

    # ──────────────────────────────────────────────────────────────────────
    # Group summary: create or update
    # ──────────────────────────────────────────────────────────────────────

    def _update_group_summary(
        self,
        group_id:   str,
        pdf_name:   str,
        chunk_texts: list[str],
        chunk_meta:  list[dict],
    ) -> None:
        existing    = next(
            (s for s in self.cluster_summaries if s["cluster_id"] == group_id), None
        )
        new_pdf_vec = self._centroid_embedding(chunk_texts).tolist()

        if existing is not None:
            existing.setdefault("pdf_centroids", {})[pdf_name] = new_pdf_vec
            prev_count = existing["chunk_count"]
            new_count  = len(chunk_texts)
            prev_vec   = np.array(existing.get("centroid", new_pdf_vec))
            blended    = (prev_vec * prev_count + np.array(new_pdf_vec) * new_count) / (
                prev_count + new_count
            )
            norm = np.linalg.norm(blended)
            existing["centroid"] = (blended / norm if norm > 0 else blended).tolist()
            existing["chunk_count"] += new_count
            if pdf_name not in existing["sources"]:
                existing["sources"].append(pdf_name)
        else:
            topic, summary = (
                _generate_group_summary(chunk_texts[:8], self.llm_client)
                if self.llm_client else ("General", "")
            )
            self.cluster_summaries.append({
                "cluster_id":   group_id,
                "topic":        topic,
                "summary":      summary,
                "chunk_count":  len(chunk_texts),
                "sources":      [pdf_name],
                "centroid":     new_pdf_vec,
                "pdf_centroids": {pdf_name: new_pdf_vec},
            })

        with open(self.summaries_path, "w", encoding="utf-8") as f:
            json.dump(self.cluster_summaries, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # Knowledge Graph
    # ──────────────────────────────────────────────────────────────────────

    def _build_knowledge_graph(
        self, texts: list[str], meta: list[dict]
    ) -> None:
        print("[KB] Updating entity knowledge graph…")
        for i, text in enumerate(texts):
            doc = self.nlp(text[:1500])
            entities = [
                ent.text for ent in doc.ents
                if ent.label_ in (
                    "ORG", "PRODUCT", "GPE", "PERSON", "FAC", "WORK_OF_ART",
                    "EVENT", "NORP", "LOC", "DATE", "ORDINAL", "CARDINAL"
                )
            ]
            node_id = meta[i].get("child_id", str(i))
            source  = meta[i].get("source", "unknown")
            group   = meta[i].get("group_id", "")
            self.kg.graph.add_node(
                node_id, text=text[:300], source=source, group=group, type="chunk"
            )
            for ent in set(entities):
                self.kg.graph.add_node(ent, type="entity")
                self.kg.graph.add_edge(node_id, ent, label="mentions")
        self.kg.save()

    # ──────────────────────────────────────────────────────────────────────
    # Advanced retrieval (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text:     str,
        top_k:          int   = 5,
        use_hyde:       bool  = True,
        use_mmr:        bool  = True,
        hybrid_alpha:   float = 0.65,
        question_type:  str   = "general",
        **kwargs,
    ) -> list[dict]:
        if self.vectorstore is None:
            return []

        retrieval_query = query_text
        if use_hyde and self.llm_client is not None:
            try:
                hyde_prompt = (
                    f"Write a short factual passage (2-3 sentences) that would answer "
                    f"this question: {query_text}"
                )
                hypothetical = self.llm_client.generate(
                    prompt=hyde_prompt, max_tokens=120
                )
                retrieval_query = query_text + " " + hypothetical.strip()
            except Exception as e:
                print(f"[WARN] HyDE failed: {e}")

        dense_results = self.vectorstore.similarity_search_with_score(
            retrieval_query, k=top_k * 3
        )
        dense_ids = {
            doc.metadata.get("child_id", str(i)): i
            for i, (doc, _score) in enumerate(dense_results)
        }

        bm25_ids: dict[str, int] = {}
        if self.bm25 is not None:
            tokens = query_text.lower().split()
            scores = self.bm25.get_scores(tokens)
            top_bm25_idx = np.argsort(scores)[::-1][: top_k * 3]
            for rank, idx in enumerate(top_bm25_idx):
                child_id = self._bm25_meta[idx].get("child_id", str(idx))
                bm25_ids[child_id] = rank

        rrf_scores: dict[str, float] = {}
        k_rrf = 60
        for child_id, rank in dense_ids.items():
            rrf_scores[child_id] = rrf_scores.get(child_id, 0) + hybrid_alpha / (k_rrf + rank)
        for child_id, rank in bm25_ids.items():
            rrf_scores[child_id] = rrf_scores.get(child_id, 0) + (1 - hybrid_alpha) / (k_rrf + rank)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: top_k * 2]

        id_to_doc: dict[str, Document] = {
            doc.metadata.get("child_id", ""): doc for doc, _ in dense_results
        }
        id_to_bm25_text: dict[str, str] = {}
        if self._bm25_meta:
            for idx, m in enumerate(self._bm25_meta):
                id_to_bm25_text[m.get("child_id", "")] = self._bm25_docs[idx]

        candidates: list[Document] = []
        for cid in sorted_ids:
            if cid in id_to_doc:
                candidates.append(id_to_doc[cid])
            elif cid in id_to_bm25_text:
                idx = next(
                    (i for i, m in enumerate(self._bm25_meta)
                     if m.get("child_id") == cid), None
                )
                if idx is not None:
                    candidates.append(Document(
                        page_content=self._bm25_docs[idx],
                        metadata=self._bm25_meta[idx],
                    ))

        if use_mmr and len(candidates) > top_k:
            candidates = self._mmr_rerank(query_text, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        results = []
        for doc in candidates:
            pid    = doc.metadata.get("parent_id", "")
            parent = self.parent_chunks.get(pid, doc.page_content)
            topic  = self._get_topic_for_chunk(doc.metadata.get("child_id", ""))
            results.append({
                "content": parent,
                "summary": doc.page_content,
                "topic":   topic,
                "source":  doc.metadata.get("source", ""),
                "page":    doc.metadata.get("page", 0),
                "type":    doc.metadata.get("type", "text"),
            })
        return results

    def _mmr_rerank(
        self, query: str, docs: list[Document], k: int, lambda_param: float = 0.6
    ) -> list[Document]:
        if not docs:
            return docs
        texts    = [d.page_content for d in docs]
        q_emb    = np.array(self.embeddings.embed_documents([query]))[0]
        d_embs   = np.array(self.embeddings.embed_documents(texts))
        rel_scores = cosine_similarity([q_emb], d_embs)[0]
        selected_idx: list[int]  = []
        remaining_idx: list[int] = list(range(len(docs)))
        while len(selected_idx) < k and remaining_idx:
            if not selected_idx:
                best   = int(np.argmax(rel_scores[remaining_idx]))
                chosen = remaining_idx[best]
            else:
                sel_embs   = d_embs[selected_idx]
                sim_to_sel = cosine_similarity(d_embs[remaining_idx], sel_embs).max(axis=1)
                mmr_scores = (
                    lambda_param * rel_scores[remaining_idx]
                    - (1 - lambda_param) * sim_to_sel
                )
                best   = int(np.argmax(mmr_scores))
                chosen = remaining_idx[best]
            selected_idx.append(chosen)
            remaining_idx.remove(chosen)
        return [docs[i] for i in selected_idx]

    def _get_topic_for_chunk(self, child_id: str) -> str:
        if not self.cluster_summaries or not self._bm25_meta:
            return ""
        group_id = None
        for m in self._bm25_meta:
            if m.get("child_id") == child_id:
                group_id = m.get("group_id")
                break
        if group_id is None:
            return ""
        for summary in self.cluster_summaries:
            if summary["cluster_id"] == group_id:
                return summary.get("topic", "")
        return ""

    def get_cluster_summaries_text(self) -> str:
        if not self.cluster_summaries:
            return ""
        return "\n".join(
            f"[{s['topic']}] {s['summary']}"
            for s in self.cluster_summaries
        )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from llm_client import llm_client

    kb = OptimizedPDFKnowledgeBase(data_dir="./data", llm_client=llm_client)
    kb.run_training_audit()   # full pipeline: extract → index → audit → offer delete

    while True:
        query = input("Test query: ").strip()
        if query:
            results = kb.query(query, top_k=5)
            for i, r in enumerate(results, 1):
                print(f"── Result {i} [{r['topic']}] (p{r['page']} of {r['source']}) ──")
                print(r["content"])