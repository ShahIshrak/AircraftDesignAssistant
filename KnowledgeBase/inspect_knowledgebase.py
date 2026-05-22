"""
inspect_knowledgebase.py — FAIRY Knowledge Base 3-D Graph Visualizer
=====================================================================
Single 3-D interactive graph showing every node and every connection
exactly as the query() pipeline sees them:

  Node types (Z layer):
    Topic Group  (Z=4)  — cluster from cluster_summaries.json
    PDF Source   (Z=3)  — source PDF file
    Parent Chunk (Z=2)  — large context window chunk
    Child Chunk  (Z=1)  — small retrieval unit (text or table)
    Entity       (Z=0)  — NER entity extracted by spaCy (KG)

  Edge types (matching query() retrieval path):
    group  -> pdf           (topic routing)
    pdf    -> parent        (document structure)
    parent -> child         (parent-child chunking)
    child  -> entity        (KG "mentions" relation)

  Layout:
    X, Y  — PCA(2) on FAISS embedding vectors for child nodes;
             all ancestor nodes inherit centroid of their descendants
             (position = semantic meaning, not arbitrary)
    Z     — fixed per layer

  Hover / click any node to see:
    - Node type / category
    - Topic group label
    - PDF file name
    - Page number
    - Full chunk content

Usage:
    python inspect_knowledgebase.py
    python inspect_knowledgebase.py --base "path/to/models"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE = r"C:\Users\jafri\PycharmProjects\FAIRY\KnowledgeBase\data\models"

# kind -> (legend label, hex colour, marker size)
NODE_STYLE = {
    "group":  ("Topic Group",   "#a855f7", 18),
    "pdf":    ("PDF Source",    "#f97316", 14),
    "parent": ("Parent Chunk",  "#0ea5e9",  8),
    "child":  ("Text Chunk",    "#22c55e",  5),
    "table":  ("Table Chunk",   "#eab308",  6),
    "entity": ("Entity (KG)",   "#ef4444",  5),
}

# edge kind -> (legend label, hex colour, line width)
EDGE_STYLE = {
    "group_to_pdf":    ("Group -> PDF",         "#a855f7", 1.5),
    "pdf_to_parent":   ("PDF -> Parent",         "#f97316", 1.2),
    "parent_to_child": ("Parent -> Child",       "#0ea5e9", 0.8),
    "pdf_to_child":    ("PDF -> Child (direct)", "#f97316", 0.6),
    "chunk_to_entity": ("Chunk -> Entity",       "#ef4444", 0.6),
}

LAYER_Z = {
    "group": 9.0, "pdf": 7.0, "parent": 4.0,
    "child": 1.0, "table": 1.0, "entity": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_kg(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get("graph", nx.Graph())
    return data.graph if hasattr(data, "graph") else data


def load_faiss_docstore(faiss_path: str, pkl_path: str):
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        ds_data = pickle.load(f)

    def _raw(obj):
        if hasattr(obj, "_dict"):
            return obj._dict
        if hasattr(obj, "docstore") and hasattr(obj.docstore, "_dict"):
            return obj.docstore._dict
        return None

    def _is_int_map(obj):
        if not isinstance(obj, dict):
            return False
        sample = next(iter(obj), None)
        return sample is None or isinstance(sample, int)

    if isinstance(ds_data, tuple) and len(ds_data) == 2:
        a, b = ds_data
        ra, rb = _raw(a), _raw(b)
        if ra is not None and _is_int_map(b):
            raw, idx_map = ra, b
        elif rb is not None and _is_int_map(a):
            raw, idx_map = rb, a
        elif ra is not None:
            raw, idx_map = ra, {}
        elif rb is not None:
            raw, idx_map = rb, {}
        else:
            raw    = a if isinstance(a, dict) else (b if isinstance(b, dict) else {})
            idx_map = {}
        doc_items = [(idx_map.get(i, str(i)), raw.get(idx_map.get(i, ""), None))
                     for i in range(index.ntotal)]
    else:
        raw = _raw(ds_data) or (ds_data if isinstance(ds_data, dict) else {})
        doc_items = list(raw.items())

    return index, doc_items


def load_bm25(path: str):
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    return p.get("docs", []), p.get("meta", [])


def load_summaries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(kg, doc_items, bm25_docs, bm25_meta,
                summaries, parent_chunks) -> nx.DiGraph:
    G = nx.DiGraph()

    # Pre-compute lookup tables
    group_of_src: dict[str, str] = {}
    group_topic:  dict[str, str] = {}
    group_summary: dict[str, str] = {}
    for s in summaries:
        gid = s["cluster_id"]
        group_topic[gid]   = s.get("topic", gid)
        group_summary[gid] = s.get("summary", "")
        for src in s.get("sources", []):
            group_of_src[src] = gid

    # Topic group nodes
    for s in summaries:
        gid = s["cluster_id"]
        G.add_node(gid,
                   kind="group",
                   label=group_topic[gid],
                   topic=group_topic[gid],
                   pdf="—",
                   page="—",
                   content=group_summary[gid],
                   chunk_count=s.get("chunk_count", 0))

    # PDF nodes
    for src in {m.get("source", "") for m in bm25_meta if m.get("source")}:
        gid = group_of_src.get(src, "")
        G.add_node(src,
                   kind="pdf",
                   label=src,
                   topic=group_topic.get(gid, "—"),
                   pdf=src,
                   page="—",
                   content=f"Source PDF: {src}")
        if gid and G.has_node(gid):
            G.add_edge(gid, src, kind="group_to_pdf")

    # Parent chunk nodes
    for pid, text in parent_chunks.items():
        parts = pid.split("__")
        src  = parts[0] if parts else "unknown"
        page = parts[1].lstrip("p") if len(parts) > 1 else "—"
        gid  = group_of_src.get(src, "")
        G.add_node(pid,
                   kind="parent",
                   label=f"{src}  p{page}",
                   topic=group_topic.get(gid, "—"),
                   pdf=src,
                   page=page,
                   content=text)
        if G.has_node(src):
            G.add_edge(src, pid, kind="pdf_to_parent")

    # Child / table nodes
    for i, meta in enumerate(bm25_meta):
        cid  = meta.get("child_id", str(i))
        kind = "table" if meta.get("type") == "table" else "child"
        src  = meta.get("source", "")
        pid  = meta.get("parent_id", "")
        page = str(meta.get("page", "—"))
        gid  = group_of_src.get(src, "")
        text = bm25_docs[i] if i < len(bm25_docs) else ""
        G.add_node(cid,
                   kind=kind,
                   label=f"{src}  p{page}",
                   topic=group_topic.get(gid, "—"),
                   pdf=src,
                   page=page,
                   content=text)
        if pid and G.has_node(pid):
            G.add_edge(pid, cid, kind="parent_to_child")
        elif src and G.has_node(src):
            G.add_edge(src, cid, kind="pdf_to_child")

    # Entity nodes  +  chunk->entity edges
    for n, attr in kg.nodes(data=True):
        if attr.get("type") == "entity":
            G.add_node(n,
                       kind="entity",
                       label=str(n),
                       topic="—",
                       pdf=attr.get("source", "—"),
                       page="—",
                       content=f"Named entity: {n}")
    for u, v in kg.edges():
        if (kg.nodes[u].get("type") != "entity"
                and kg.nodes[v].get("type") == "entity"
                and G.has_node(u) and G.has_node(v)):
            G.add_edge(u, v, kind="chunk_to_entity")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Layout  (O(N), no simulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_positions(G: nx.DiGraph, embeddings, doc_items) -> dict:
    rng = np.random.default_rng(42)
    pos: dict[str, np.ndarray] = {}

    # child_id -> FAISS row
    cid_idx: dict[str, int] = {}
    for i, (uid, doc) in enumerate(doc_items):
        if doc is not None and hasattr(doc, "metadata"):
            cid = doc.metadata.get("child_id", "")
            if cid:
                cid_idx[cid] = i

    # PCA(2) on FAISS embeddings -> semantic XY for child/table
    pca_xy: dict[str, np.ndarray] = {}
    if embeddings is not None and embeddings.shape[0] >= 2:
        reduced = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        for axis in range(2):
            col  = reduced[:, axis]
            span = col.max() - col.min()
            if span > 0:
                reduced[:, axis] = (col - col.min()) / span * 16 - 8
        for cid, idx in cid_idx.items():
            if idx < len(reduced):
                pca_xy[cid] = reduced[idx]

    def rand_xy():
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, 8)
        return np.array([r * np.cos(a), r * np.sin(a)])

    # 1. child / table
    for n, attr in G.nodes(data=True):
        k = attr.get("kind")
        if k not in ("child", "table"):
            continue
        xy = pca_xy.get(n, rand_xy())
        pos[n] = np.array([xy[0], xy[1], LAYER_Z[k]])

    # 2. parent = centroid of its children
    p2c: dict = defaultdict(list)
    for u, v, e in G.edges(data=True):
        if e.get("kind") == "parent_to_child" and v in pos:
            p2c[u].append(pos[v][:2])
    for n, attr in G.nodes(data=True):
        if attr.get("kind") != "parent":
            continue
        kids = p2c.get(n)
        xy   = np.mean(kids, axis=0) if kids else rand_xy()
        pos[n] = np.array([xy[0], xy[1], LAYER_Z["parent"]])

    # 3. pdf = centroid of its parents
    p2par: dict = defaultdict(list)
    for u, v, e in G.edges(data=True):
        if e.get("kind") == "pdf_to_parent" and v in pos:
            p2par[u].append(pos[v][:2])
    for n, attr in G.nodes(data=True):
        if attr.get("kind") != "pdf":
            continue
        pars = p2par.get(n)
        xy   = np.mean(pars, axis=0) if pars else rand_xy()
        pos[n] = np.array([xy[0], xy[1], LAYER_Z["pdf"]])

    # 4. group = centroid of its PDFs
    g2pdf: dict = defaultdict(list)
    for u, v, e in G.edges(data=True):
        if e.get("kind") == "group_to_pdf" and v in pos:
            g2pdf[u].append(pos[v][:2])
    for n, attr in G.nodes(data=True):
        if attr.get("kind") != "group":
            continue
        pdfs = g2pdf.get(n)
        xy   = np.mean(pdfs, axis=0) if pdfs else rand_xy()
        pos[n] = np.array([xy[0], xy[1], LAYER_Z["group"]])

    # 5. entity = orbit around centroid of citing chunks
    e2c: dict = defaultdict(list)
    for u, v, e in G.edges(data=True):
        if e.get("kind") == "chunk_to_entity" and u in pos:
            e2c[v].append(pos[u][:2])
    ents   = [n for n, a in G.nodes(data=True) if a.get("kind") == "entity"]
    n_ents = max(len(ents), 1)
    for i, n in enumerate(ents):
        angle = (i / n_ents) * 2 * np.pi
        cites = e2c.get(n)
        if cites:
            ctr = np.mean(cites, axis=0)
            xy  = ctr + np.array([np.cos(angle), np.sin(angle)]) * 1.5
        else:
            xy = np.array([9 * np.cos(angle), 9 * np.sin(angle)])
        pos[n] = np.array([xy[0], xy[1], LAYER_Z["entity"]])

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Hover text
# ─────────────────────────────────────────────────────────────────────────────

def make_hover(attr: dict) -> str:
    kind    = attr.get("kind", "?")
    nlabel  = NODE_STYLE.get(kind, (kind, "", 0))[0]
    topic   = attr.get("topic",   "—")
    pdf     = attr.get("pdf",     "—")
    page    = attr.get("page",    "—")
    content = str(attr.get("content", "")).strip()

    # Wrap content at ~90 chars per line
    lines_c = []
    for i in range(0, min(len(content), 700), 90):
        lines_c.append(content[i:i + 90])
    if len(content) > 700:
        lines_c.append("…(truncated)")
    content_html = "<br>".join(lines_c)

    parts = [
        f"<b>Type:</b> {nlabel}",
        f"<b>Category / Topic:</b> {topic}",
        f"<b>PDF:</b> {pdf}",
    ]
    if str(page) not in ("—", ""):
        parts.append(f"<b>Page:</b> {page}")
    if attr.get("chunk_count"):
        parts.append(f"<b>Chunks in group:</b> {attr['chunk_count']}")
    if content_html:
        parts.append(f"<b>Content:</b><br>{content_html}")

    return "<br>".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(G: nx.DiGraph, pos: dict) -> go.Figure:
    fig = go.Figure()

    # Which layer does each edge kind's SOURCE node belong to?
    # Used to hide edges together with their source layer.
    EDGE_LAYER = {
        "group_to_pdf":    "Topic Groups",
        "pdf_to_parent":   "PDFs",
        "parent_to_child": "Parent Chunks",
        "pdf_to_child":    "PDFs",
        "chunk_to_entity": "Child/Table Chunks",
    }

    # Ordered layers: (display name, node kinds that belong to it)
    LAYERS: list[tuple[str, list[str]]] = [
        ("Topic Groups",       ["group"]),
        ("PDFs",               ["pdf"]),
        ("Parent Chunks",      ["parent"]),
        ("Child/Table Chunks", ["child", "table"]),
        ("Entities (KG)",      ["entity"]),
    ]
    KIND_TO_LAYER = {k: lname for lname, kinds in LAYERS for k in kinds}
    layer_names   = [lname for lname, _ in LAYERS]

    # We record which layer each trace belongs to as we add them.
    trace_layer: list[str] = []

    # ── edges ─────────────────────────────────────────────────────────────────
    ex: dict[str, list] = defaultdict(list)
    ey: dict[str, list] = defaultdict(list)
    ez: dict[str, list] = defaultdict(list)

    for u, v, eattr in G.edges(data=True):
        ek = eattr.get("kind", "parent_to_child")
        if ek not in EDGE_STYLE or u not in pos or v not in pos:
            continue
        p0, p1 = pos[u], pos[v]
        ex[ek] += [float(p0[0]), float(p1[0]), None]
        ey[ek] += [float(p0[1]), float(p1[1]), None]
        ez[ek] += [float(p0[2]), float(p1[2]), None]

    for ek, (ename, ecolor, ewidth) in EDGE_STYLE.items():
        if not ex[ek]:
            continue
        fig.add_trace(go.Scatter3d(
            x=ex[ek], y=ey[ek], z=ez[ek],
            mode="lines",
            name=ename,
            line=dict(width=ewidth, color=ecolor),
            hoverinfo="none",
            opacity=0.30,
            legendgroup="edge_" + ek,
        ))
        trace_layer.append(EDGE_LAYER.get(ek, "Child/Table Chunks"))

    # ── nodes ─────────────────────────────────────────────────────────────────
    kind_buckets: dict[str, list] = defaultdict(list)
    for n, attr in G.nodes(data=True):
        kind_buckets[attr.get("kind", "child")].append((n, attr))

    for kind, bucket in kind_buckets.items():
        nlabel, color, size = NODE_STYLE.get(kind, (kind, "#888", 5))
        xs, ys, zs, hovers, txt = [], [], [], [], []

        for n, attr in bucket:
            if n not in pos:
                continue
            p = pos[n]
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
            hovers.append(make_hover(attr))
            short = attr.get("label", str(n))
            txt.append(short[:40] if kind in ("group", "pdf") else "")

        show_text = kind in ("group", "pdf")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text" if show_text else "markers",
            name=nlabel,
            marker=dict(
                size=size,
                color=color,
                opacity=0.95 if kind in ("group", "pdf") else 0.72,
                line=dict(width=0.5, color="#0d0d12"),
                symbol="diamond" if kind == "group" else "circle",
            ),
            text=txt if show_text else None,
            textposition="top center",
            textfont=dict(size=9, color=color),
            customdata=hovers,
            hovertemplate="%{customdata}<extra></extra>",
            legendgroup=kind,
        ))
        trace_layer.append(KIND_TO_LAYER.get(kind, kind))

    n_traces = len(fig.data)

    # ── Visibility helpers ────────────────────────────────────────────────────

    def _vis(hidden: set[str]) -> list[bool]:
        """Visibility list with the given layer names hidden."""
        return [trace_layer[i] not in hidden for i in range(n_traces)]

    # ── Build controls ────────────────────────────────────────────────────────
    #
    # Layout (top of figure, two rows):
    #
    #   ROW A  "Solo:"   [dropdown — show exactly one layer]
    #
    #   ROW B  "Toggle:" [Topic Groups] [PDFs] [Parent Chunks]
    #                    [Child/Table Chunks] [Entities (KG)]  | [Reset]
    #
    # Toggle buttons use args / args2 so each one independently flips
    # between "hide this layer" and "show this layer" on repeated clicks.
    # This is the only way to get true independent multi-layer toggling
    # in a static Plotly HTML figure without JavaScript callbacks.
    #
    # args  = what happens on the FIRST click  (hide the layer)
    # args2 = what happens on the SECOND click (restore full visibility)
    #
    # Caveat: args2 restores *all* layers because Plotly's restyle can't
    # read the current visibility state of other traces.  For a workflow
    # where you want, e.g., PDFs + Entities hidden simultaneously, use
    # the Solo dropdown to get a clean single-layer view, then use the
    # toggle buttons to incrementally peel layers off from "Show All".

    btn_style = dict(
        bgcolor="#1e293b",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(color="#cbd5e1", size=11),
    )

    # ── Row A: Solo dropdown ──────────────────────────────────────────────────
    solo_buttons = []
    for lname in layer_names:
        others = {n for n in layer_names if n != lname}
        solo_buttons.append(dict(
            label=f"Only: {lname}",
            method="restyle",
            args=[{"visible": _vis(others)}],
        ))
    solo_buttons.append(dict(
        label="Show All",
        method="restyle",
        args=[{"visible": [True] * n_traces}],
    ))

    # ── Row B: one toggle button per layer ────────────────────────────────────
    # Each is wrapped in its own single-button "buttons" updatemenus entry
    # so they sit side-by-side at the same y position and each maintains
    # its own args/args2 toggle state independently.
    toggle_xs = [0.01, 0.16, 0.31, 0.46, 0.61]   # x anchors for 5 buttons

    toggle_menus = []
    for i, lname in enumerate(layer_names):
        _, color, _ = NODE_STYLE.get(
            # pick the first node kind that maps to this layer
            next(k for k, ln in KIND_TO_LAYER.items() if ln == lname),
            ("", "#cbd5e1", 0)
        )
        toggle_menus.append(dict(
            type="buttons",
            direction="right",
            x=toggle_xs[i],
            y=1.12,
            xanchor="left",
            showactive=True,
            active=0,           # 0 = first state (args = hide)
            buttons=[dict(
                label=lname,
                method="restyle",
                # First click  → hide this layer
                args =[{"visible": _vis({lname})}],
                # Second click → show everything again
                args2=[{"visible": [True] * n_traces}],
            )],
            pad=dict(r=4, t=3),
            **btn_style,
        ))

    # Reset button (far right of Row B)
    reset_menu = dict(
        type="buttons",
        direction="right",
        x=0.78,
        y=1.12,
        xanchor="left",
        showactive=False,
        buttons=[dict(
            label="↺ Reset",
            method="restyle",
            args=[{"visible": [True] * n_traces}],
        )],
        pad=dict(r=4, t=3),
        **btn_style,
    )

    # Solo dropdown (Row A)
    solo_menu = dict(
        type="dropdown",
        direction="down",
        x=0.01,
        y=1.24,
        xanchor="left",
        showactive=True,
        active=len(solo_buttons) - 1,   # default = "Show All"
        buttons=solo_buttons,
        pad=dict(r=6, t=4),
        **btn_style,
    )

    # ── layout ────────────────────────────────────────────────────────────────
    layer_ticks = [
        (4.0, "Topic Groups"),
        (3.0, "PDFs"),
        (2.0, "Parent Chunks"),
        (1.0, "Child / Table Chunks"),
        (0.0, "Entities (KG)"),
    ]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07090f",
        title=dict(
            text="FAIRY Knowledge Base  —  3-D Semantic Graph",
            font=dict(size=21, color="#e2e8f0"),
            x=0.5,
        ),
        scene=dict(
            bgcolor="#0b0d14",
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(
                title=dict(text="Layer", font=dict(color="#94a3b8", size=12)),
                tickvals=[z for z, _ in layer_ticks],
                ticktext=[lbl for _, lbl in layer_ticks],
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="#1e293b",
                zerolinecolor="#334155",
            ),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.75)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        legend=dict(
            bgcolor="#111827",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
            itemsizing="constant",
            tracegroupgap=4,
            x=1.02, y=0.5,
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(size=12, color="#f1f5f9"),
            align="left",
            namelength=0,
        ),
        updatemenus=[solo_menu] + toggle_menus + [reset_menu],
        annotations=[
            dict(text="<b>Solo:</b>",
                 x=0.01, y=1.285, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="#94a3b8", size=10),
                 xanchor="left"),
            dict(text="<b>Toggle</b> (click twice to restore):",
                 x=0.01, y=1.165, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="#94a3b8", size=10),
                 xanchor="left"),
        ],
        margin=dict(l=0, r=120, b=0, t=120),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FAIRY KB 3-D Visualizer")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Path to the models/ directory")
    args = parser.parse_args()
    base = Path(args.base)

    paths = {
        "kg":       base / "knowledge_graph.pkl",
        "faiss":    base / "faiss_index" / "index.faiss",
        "pkl":      base / "faiss_index" / "index.pkl",
        "bm25":     base / "bm25_corpus.json",
        "summaries":base / "cluster_summaries.json",
        "parents":  base / "parent_chunks.json",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("[1/6] Knowledge Graph…")
    kg = load_kg(str(paths["kg"]))

    print("[2/6] FAISS index + docstore…")
    faiss_index, doc_items = load_faiss_docstore(
        str(paths["faiss"]), str(paths["pkl"]))
    print(f"      {faiss_index.ntotal} vectors")

    print("[3/6] BM25 corpus…")
    bm25_docs, bm25_meta = load_bm25(str(paths["bm25"]))
    print(f"      {len(bm25_docs)} documents")

    print("[4/6] Cluster summaries…")
    summaries = load_summaries(str(paths["summaries"]))
    print(f"      {len(summaries)} topic groups")

    print("[5/6] Parent chunks…")
    parents = load_parents(str(paths["parents"]))
    print(f"      {len(parents)} parent chunks")

    print("[6/6] Building graph + layout…")
    G = build_graph(kg, doc_items, bm25_docs, bm25_meta, summaries, parents)
    print(f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    cap = min(faiss_index.ntotal, 50_000)
    try:
        embeddings = faiss_index.reconstruct_n(0, cap)
    except Exception as e:
        print(f"      [WARN] Embeddings unavailable: {e}")
        embeddings = None

    pos = compute_positions(G, embeddings, doc_items)

    print("Rendering…  (hover any node for full details)\n")
    fig = plot_3d(G, pos)
    fig.show()


if __name__ == "__main__":
    main()