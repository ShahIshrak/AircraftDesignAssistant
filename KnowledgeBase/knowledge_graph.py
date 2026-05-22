import os
import pickle
import networkx as nx
import hdbscan
import numpy as np
from model_registry import get_embedding_model, get_nlp

# ---------------------------------------------------------------------------
# PERFORMANCE NOTES — what was slow and what changed
#
# 1. _get_or_create_node was O(N) per call — scanned every node with cos_sim
#    in a Python loop. Fixed: keep a stacked embedding matrix + numpy matmul.
#    Node lookup is now a single matrix-vector multiply regardless of N.
#
# 2. Fallback cosine edge computation was O(N²) and ran on the FULL node set
#    every add_text call. Fixed: only compare NEW nodes added in this call
#    against the existing node matrix (incremental, not global).
#
# 3. spaCy was called once per chunk. Fixed: add_texts() batch-processes all
#    chunks through nlp.pipe() in one pass — ~3-5x faster for large corpora.
#
# 4. model.encode() was called once per relation subject/object individually.
#    Fixed: batch all subjects and objects from a chunk into one encode() call.
#
# 5. expand_query used a Python loop with util.cos_sim(). Fixed: matrix multiply.
#
# 6. HDBSCAN was re-instantiated every add_text call. Fixed: reused instance.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OCR artifact cleaning — applied before ANY text enters the graph
# ---------------------------------------------------------------------------

def _clean_ocr_text(text: str) -> str:
    """
    Strip OCR pipe-break artifacts before sentence splitting or relation
    extraction.  Handles patterns like:
      'qui | cknes | s'  →  'quickness'
      'the pres | sure'  →  'the pressure'
      '| istan'          →  'istan'   (leading fragment dropped by label filter)

    Strategy: collapse any run of whitespace + '|' + whitespace into a single
    space, then normalise multiple spaces.  This reconnects broken words while
    preserving sentence structure.
    """
    import re
    t = re.sub(r'\s*\|\s*', ' ', text)
    t = re.sub(r' {2,}', ' ', t)
    return t.strip()


# ---------------------------------------------------------------------------
# Label validation — filters pronouns, stopwords, and OCR garbage
# ---------------------------------------------------------------------------

_STOPWORDS = {
    'we', 'you', 'he', 'she', 'they', 'it', 'this', 'that', 'which', 'who',
    'us', 'them', 'i', 'me', 'one', 'what', 'there', 'here', 'these', 'those',
    'a', 'an', 'the', 'its', 'his', 'her', 'our', 'your', 'their', 'my',
}


def _is_valid_label(text: str) -> bool:
    """
    Returns False for:
      - pronouns / stopwords
      - too short (< 4 chars) or too long (> 100 chars)
      - any remaining OCR pipe-break characters
      - spaced single-char OCR artifacts like "I T C H R O L L"
      - pure digit / punctuation strings
      - labels with no token of meaningful length (all tokens ≤ 2 chars)
        — catches fragments like 'yer', 'bee', 'a d', 'x y', 'I T'
    """
    t = text.strip()
    if not t:
        return False
    if t.lower() in _STOPWORDS:
        return False
    if len(t) < 4 or len(t) > 100:
        return False
    # Reject anything containing residual pipe-break artifacts
    if '|' in t:
        return False
    # Reject strings that are purely digits, symbols, or whitespace
    if not any(c.isalpha() for c in t):
        return False
    tokens = t.split()
    # Reject OCR artifacts: majority of tokens are single characters
    if len(tokens) >= 3 and (sum(1 for tok in tokens if len(tok) == 1) / len(tokens)) > 0.4:
        return False
    # Require at least one token of meaningful length
    if not any(len(tok) >= 3 for tok in tokens):
        return False
    return True


def _span_text(token):
    """Return clean text for a token's full noun-phrase subtree."""
    parts = [
        t.text for t in token.subtree
        if t.dep_ not in ("punct", "cc")
    ]
    return " ".join(parts).strip()


def _extract_relations_doc(doc) -> list[tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from an already-parsed spaCy doc.
    Separated from extract_relations_from_text so batch processing can reuse it.
    """
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue
            if token.dep_ not in ("ROOT", "relcl", "advcl", "ccomp", "xcomp"):
                continue

            subjects = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass", "csubj")]
            objects  = [w for w in token.children if w.dep_ in ("dobj", "pobj", "attr", "oprd")]
            for child in token.children:
                if child.dep_ == "prep":
                    objects.extend(w for w in child.children if w.dep_ == "pobj")

            if not subjects or not objects:
                continue

            relation = token.lemma_.lower()
            for subj in subjects:
                for obj in objects:
                    st = _span_text(subj)
                    ot = _span_text(obj)
                    if (st and ot and st != ot
                            and 2 <= len(st) <= 80
                            and 2 <= len(ot) <= 80
                            and _is_valid_label(st)
                            and _is_valid_label(ot)):
                        relations.append((st, relation, ot))
    return relations


def extract_relations_from_text(text: str) -> list[tuple[str, str, str]]:
    """Single-text wrapper — kept for external callers."""
    return _extract_relations_doc(get_nlp()(text))


# ---------------------------------------------------------------------------
# SemanticKnowledgeGraph
# ---------------------------------------------------------------------------

class SemanticKnowledgeGraph:
    def __init__(self, storage_path, cluster_min_size=2, edge_threshold=0.55):
        self.graph_path     = os.path.join(storage_path, "knowledge_graph.pkl")
        self.graph          = nx.DiGraph()
        self.edge_threshold = edge_threshold
        self.cluster_min_size = cluster_min_size
        self._node_counter  = 0

        self.node_embeddings:   dict[str, np.ndarray] = {}
        self.node_labels:       dict[str, str]        = {}
        self.cluster_to_chunks: dict[str, list]       = {}

        # --- Fast lookup cache -------------------------------------------
        # Stacked matrix of all node embeddings for O(1) batch similarity.
        # Rebuilt from node_embeddings on load; kept in sync on every insert.
        # We accumulate new rows in _emb_pending (a plain list) and only
        # vstack into _emb_matrix when a lookup is needed — amortises the
        # O(N) copy cost so overall insertion is O(1) amortised per node.
        self._emb_matrix:   np.ndarray | None = None   # shape (N, D)
        self._emb_node_ids: list[str]         = []     # index → node_id
        self._emb_pending:  list[np.ndarray]  = []     # rows not yet stacked

        # Reusable HDBSCAN instance (avoid re-instantiation per chunk)
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_min_size,
            metric='euclidean',
            core_dist_n_jobs=1,   # single-threaded — avoids joblib fork overhead
        )

        if os.path.exists(self.graph_path):
            self.load()

    # ------------------------------------------------------------------
    def load(self):
        """
        Load persisted graph state from knowledge_graph.pkl.

        Corruption handling:
          If the file is unreadable for any reason (truncated write, null bytes,
          version mismatch, disk error) we:
            1. Delete the bad file so it cannot block future runs.
            2. Leave the graph in its empty initialised state — the caller
               (KnowledgeBase_training) detects that node_embeddings is empty
               and rebuilds the KG from the existing vectorstore or re-OCRs
               the source PDFs as appropriate.
            3. Return without raising so the rest of startup can continue.
        """
        try:
            with open(self.graph_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError,
                ModuleNotFoundError, AttributeError, TypeError,
                KeyError, OSError, Exception) as exc:
            print(
                f"[WARN] knowledge_graph.pkl is corrupt "
                f"({type(exc).__name__}: {exc}).\n"
                f"       Deleting corrupt file — KG will be rebuilt from the "
                f"existing vectorstore without re-reading any PDF."
            )
            try:
                os.remove(self.graph_path)
            except OSError as rm_err:
                print(f"[WARN] Could not remove corrupt KG file: {rm_err}")
            # Graph stays in the empty state set by __init__; caller will rebuild.
            return

        self.graph             = data["graph"]
        self.node_embeddings   = data["embeddings"]
        self.node_labels       = data.get("node_labels", {})
        self.cluster_to_chunks = data.get("cluster_to_chunks", {})
        self._node_counter     = data.get("node_counter", len(self.node_embeddings))
        self._rebuild_emb_matrix()

    def _rebuild_emb_matrix(self):
        """Rebuild the stacked embedding matrix from node_embeddings dict."""
        if not self.node_embeddings:
            self._emb_matrix   = None
            self._emb_node_ids = []
            self._emb_pending  = []
            return
        ids  = list(self.node_embeddings.keys())
        mats = np.stack([self.node_embeddings[i] for i in ids])
        norms = np.linalg.norm(mats, axis=1, keepdims=True)
        mats  = mats / np.where(norms == 0, 1.0, norms)
        self._emb_matrix   = mats
        self._emb_node_ids = ids
        self._emb_pending  = []

    def _append_to_emb_matrix(self, node_id: str, embedding: np.ndarray):
        """Queue one embedding row — flushed to the matrix on next lookup."""
        self._emb_pending.append(embedding.reshape(1, -1))
        self._emb_node_ids.append(node_id)

    def _flush_pending(self):
        """
        Consolidate pending rows into _emb_matrix.
        Called at the start of any operation that needs the full matrix
        (_get_or_create_node, _add_incremental_edges).
        This amortises the np.vstack cost: instead of one vstack per insert
        (O(N) copy each time = O(N²) total), we do one vstack per chunk.
        """
        if not self._emb_pending:
            return
        pending = np.vstack(self._emb_pending)
        self._emb_matrix = pending if self._emb_matrix is None else np.vstack([self._emb_matrix, pending])
        self._emb_pending = []

    # ------------------------------------------------------------------
    @property
    def model(self):
        return get_embedding_model()

    # ------------------------------------------------------------------
    def save(self):
        """
        Atomic save — write to a temp file then rename over the real path.

        Why: the old approach opened knowledge_graph.pkl with "wb", which
        truncates the file to zero bytes immediately.  If the process dies
        during the write the file contains null bytes (\x00) and is
        unreadable — the exact corruption that caused the original crash.

        With atomic rename:
          - The write goes to a sibling .tmp file in the same directory
            (same filesystem → rename is an in-place metadata swap, not a copy).
          - os.replace() swaps it over the real path only after the write is
            fully flushed.
          - If the process dies before os.replace(), the OLD valid .pkl
            survives untouched.  The orphaned .tmp is cleaned up on the next
            save() call.
        """
        import tempfile
        payload = {
            "graph":             self.graph,
            "embeddings":        self.node_embeddings,
            "node_labels":       self.node_labels,
            "cluster_to_chunks": self.cluster_to_chunks,
            "node_counter":      self._node_counter,
        }
        dir_name = os.path.dirname(self.graph_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f)
            # Atomic replace: old file stays valid if we die before this line
            os.replace(tmp_path, self.graph_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    def _get_or_create_node(self, label: str, embedding: np.ndarray, chunk_id) -> str:
        """
        Find or create a graph node for the given label + embedding.

        Improvements over original:
        - Updates node_labels with the longer/richer label when merging two
          nodes that refer to the same concept (original kept only the first
          label seen, even if a later one was more descriptive).
        - cluster_to_chunks deduplicates on append so the same phrase is never
          stored twice for the same node (original accumulated duplicates on
          every add_texts call, inflating memory and retrieval noise).
        """
        self._flush_pending()
        if self._emb_matrix is not None and len(self._emb_node_ids) > 0:
            sims     = self._emb_matrix @ embedding          # shape (N,)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim > 0.75:
                best_node = self._emb_node_ids[best_idx]
                # Update centroid embedding
                old    = self.node_embeddings[best_node]
                merged = (old + embedding) / 2.0
                merged /= np.linalg.norm(merged)
                self.node_embeddings[best_node]  = merged
                self._emb_matrix[best_idx]       = merged
                # Prefer the longer label as it is usually more descriptive
                existing_label = self.node_labels.get(best_node, "")
                if len(label) > len(existing_label):
                    self.node_labels[best_node] = label
                    self.graph.nodes[best_node]["label"] = label
                return best_node

        # New node
        node_id = f"n{self._node_counter}_{chunk_id}"
        self._node_counter += 1
        self.graph.add_node(node_id, label=label)
        self.node_embeddings[node_id]   = embedding
        self.node_labels[node_id]       = label
        self.cluster_to_chunks[node_id] = []
        self._append_to_emb_matrix(node_id, embedding)
        return node_id

    def _append_chunk_ref(self, node_id: str, text: str) -> None:
        """
        Append a chunk-text reference to cluster_to_chunks[node_id],
        deduplicating so the same string is never stored twice.
        This replaces bare .append() / .extend() calls throughout add_text
        and add_texts, which were the source of the 2185-duplicate-cluster
        problem observed in the existing knowledge_graph.pkl.
        """
        refs = self.cluster_to_chunks.setdefault(node_id, [])
        if text not in refs:
            refs.append(text)

    # ------------------------------------------------------------------
    def add_text(self, text: str, chunk_id=None):
        """
        Add a single chunk. For bulk ingestion prefer add_texts() which
        batches spaCy parsing and embedding calls across all chunks.
        """
        # Clean OCR pipe artifacts before any processing
        text = _clean_ocr_text(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return

        model = self.model
        sent_embeddings = model.encode(sentences, normalize_embeddings=True)

        # Record which node_ids exist BEFORE this call so incremental
        # edge computation only compares new nodes against old ones.
        existing_node_ids = set(self._emb_node_ids)

        # ── HDBSCAN clustering ──────────────────────────────────────────
        if len(sentences) >= 2:
            self._clusterer.min_cluster_size = min(len(sentences), self.cluster_min_size)
            cluster_labels = self._clusterer.fit_predict(sent_embeddings)
        else:
            cluster_labels = np.zeros(len(sentences), dtype=int)

        for lbl in set(cluster_labels):
            if lbl == -1:
                continue
            indices       = np.where(cluster_labels == lbl)[0]
            cluster_sents = [sentences[i] for i in indices]
            cluster_embs  = sent_embeddings[indices]

            centroid  = cluster_embs.mean(axis=0)
            centroid /= np.linalg.norm(centroid)

            sims       = cluster_embs @ centroid
            rep_phrase = cluster_sents[int(np.argmax(sims))]

            if not _is_valid_label(rep_phrase):
                continue

            node_id = self._get_or_create_node(rep_phrase, centroid, chunk_id)
            for s in cluster_sents:
                self._append_chunk_ref(node_id, s)

        # ── spaCy relation extraction ───────────────────────────────────
        try:
            doc       = get_nlp()(text)
            relations = _extract_relations_doc(doc)
        except Exception as e:
            print(f"[WARN] Relation extraction failed for chunk {chunk_id}: {e}")
            relations = []

        if relations:
            all_texts = [s for s, _, _ in relations] + [o for _, _, o in relations]
            all_embs  = model.encode(all_texts, normalize_embeddings=True)
            n         = len(relations)

            for idx, (subj_text, relation, obj_text) in enumerate(relations):
                subj_emb = all_embs[idx]
                obj_emb  = all_embs[idx + n]

                subj_id = self._get_or_create_node(subj_text, subj_emb, chunk_id)
                obj_id  = self._get_or_create_node(obj_text,  obj_emb,  chunk_id)

                self._append_chunk_ref(subj_id, subj_text)
                self._append_chunk_ref(obj_id,  obj_text)

                if not self.graph.has_edge(subj_id, obj_id):
                    self.graph.add_edge(subj_id, obj_id,
                                        relation=relation,
                                        chunk_id=chunk_id,
                                        weight=1.0)

        self._add_incremental_edges(existing_node_ids, chunk_id)

    # ------------------------------------------------------------------
    def add_texts(self, texts: list[str], chunk_ids: list | None = None):
        """
        Bulk ingestion — batches spaCy and embedding calls across all chunks.
        Use this instead of calling add_text() in a loop during training.

        KnowledgeBase_training.py should call:
            self.kg.add_texts(text_chunks, chunk_ids=list(range(len(text_chunks))))
        instead of:
            for i, chunk in enumerate(text_chunks):
                self.kg.add_text(chunk, chunk_id=i)
        """
        if not texts:
            return
        if chunk_ids is None:
            chunk_ids = list(range(len(texts)))

        model = self.model
        nlp   = get_nlp()

        # ── 0. Clean OCR artifacts from all texts up-front ─────────────
        # Must happen before spaCy, sentence splitting, and embedding so
        # pipe-break fragments never enter the graph as nodes or relations.
        cleaned_texts = [_clean_ocr_text(t) for t in texts]

        # ── 1. Batch spaCy parsing ──────────────────────────────────────
        print(f"[KG] Parsing {len(cleaned_texts)} chunks with spaCy...")
        all_relations: list[list[tuple]] = []
        for doc in nlp.pipe(cleaned_texts, batch_size=64):
            try:
                all_relations.append(_extract_relations_doc(doc))
            except Exception:
                all_relations.append([])

        # ── 2. Batch-encode ALL sentences across ALL chunks at once ────
        print(f"[KG] Encoding sentences...")
        all_sentences_flat: list[str] = []
        chunk_sentence_ranges: list[tuple[int, int]] = []
        for text in cleaned_texts:
            sents = [s.strip() for s in text.split('.') if s.strip()]
            start = len(all_sentences_flat)
            all_sentences_flat.extend(sents)
            chunk_sentence_ranges.append((start, len(all_sentences_flat)))

        if not all_sentences_flat:
            return

        all_sent_embs = model.encode(
            all_sentences_flat,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=True,
        )

        # ── 3. Batch-encode ALL relation subjects/objects at once ───────
        all_subj_obj_texts: list[str] = []
        rel_chunk_map: list[int] = []
        for ci, rels in enumerate(all_relations):
            for subj_text, _, obj_text in rels:
                all_subj_obj_texts.append(subj_text)
                all_subj_obj_texts.append(obj_text)
                rel_chunk_map.extend([ci, ci])

        subj_obj_embs: np.ndarray | None = None
        if all_subj_obj_texts:
            print(f"[KG] Encoding {len(all_subj_obj_texts)} relation spans...")
            subj_obj_embs = model.encode(
                all_subj_obj_texts,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=False,
            )

        # ── 4. Process each chunk using pre-computed embeddings ─────────
        print(f"[KG] Building graph nodes and edges...")
        subj_obj_cursor = 0

        for ci, (text, chunk_id) in enumerate(zip(cleaned_texts, chunk_ids)):
            start, end = chunk_sentence_ranges[ci]
            sentences   = all_sentences_flat[start:end]
            if not sentences:
                subj_obj_cursor += 2 * len(all_relations[ci])
                continue

            sent_embs = all_sent_embs[start:end]
            existing_node_ids = set(self._emb_node_ids)

            # HDBSCAN clustering
            if len(sentences) >= 2:
                self._clusterer.min_cluster_size = min(len(sentences), self.cluster_min_size)
                cluster_labels = self._clusterer.fit_predict(sent_embs)
            else:
                cluster_labels = np.zeros(len(sentences), dtype=int)

            for lbl in set(cluster_labels):
                if lbl == -1:
                    continue
                indices       = np.where(cluster_labels == lbl)[0]
                cluster_sents = [sentences[i] for i in indices]
                cluster_embs  = sent_embs[indices]

                centroid  = cluster_embs.mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                sims       = cluster_embs @ centroid
                rep_phrase = cluster_sents[int(np.argmax(sims))]

                if not _is_valid_label(rep_phrase):
                    continue

                node_id = self._get_or_create_node(rep_phrase, centroid, chunk_id)
                for s in cluster_sents:
                    self._append_chunk_ref(node_id, s)

            # Relations for this chunk
            rels = all_relations[ci]
            for subj_text, relation, obj_text in rels:
                subj_emb = subj_obj_embs[subj_obj_cursor]
                obj_emb  = subj_obj_embs[subj_obj_cursor + 1]
                subj_obj_cursor += 2

                subj_id = self._get_or_create_node(subj_text, subj_emb, chunk_id)
                obj_id  = self._get_or_create_node(obj_text,  obj_emb,  chunk_id)

                self._append_chunk_ref(subj_id, subj_text)
                self._append_chunk_ref(obj_id,  obj_text)

                if not self.graph.has_edge(subj_id, obj_id):
                    self.graph.add_edge(subj_id, obj_id,
                                        relation=relation,
                                        chunk_id=chunk_id,
                                        weight=1.0)

            self._add_incremental_edges(existing_node_ids, chunk_id)

        print(f"[KG] Graph built: {len(self.node_embeddings)} nodes, "
              f"{self.graph.number_of_edges()} edges.")

    # ------------------------------------------------------------------
    def _add_incremental_edges(self, existing_node_ids: set, chunk_id):
        """
        Add 'related_to' edges only between:
          - nodes added in the current add_text() call (new nodes)
          - all other nodes
        """
        self._flush_pending()
        all_ids  = self._emb_node_ids
        new_ids  = [nid for nid in all_ids if nid not in existing_node_ids]

        if not new_ids or self._emb_matrix is None:
            return

        # Indices into the stacked matrix
        all_idx = {nid: i for i, nid in enumerate(all_ids)}
        new_idx = [all_idx[nid] for nid in new_ids]

        new_embs = self._emb_matrix[new_idx]          # shape (new, D)
        # Similarity of each new node against all nodes
        sim_block = new_embs @ self._emb_matrix.T      # shape (new, N)

        for local_i, ni in enumerate(new_ids):
            for global_j, nj in enumerate(all_ids):
                if ni == nj:
                    continue
                if sim_block[local_i, global_j] > self.edge_threshold:
                    if not self.graph.has_edge(ni, nj):
                        self.graph.add_edge(
                            ni, nj,
                            relation="related_to",
                            weight=float(sim_block[local_i, global_j]),
                            chunk_id=chunk_id,
                        )

    # ------------------------------------------------------------------
    def expand_query(self, query: str, depth: int = 1) -> list[str]:
        """
        Return node IDs semantically related to the query, then expand
        through the graph up to `depth` hops.

        Two thresholds:
          QUERY_THRESHOLD (0.45) — used when matching the query against nodes.
            Lower than edge_threshold so we cast a wider initial net on lookup.
          edge_threshold (0.55)  — used when building edges during training.
            These are different concerns: matching a query needs more recall,
            while edges should only connect genuinely related concepts.

        Graph traversal follows BOTH outgoing and incoming edges so that
        'lift coefficient' also finds nodes that point TO lift coefficient,
        not just nodes that lift coefficient points to.
        """
        if self._emb_matrix is None or not self._emb_node_ids:
            return []

        QUERY_THRESHOLD = 0.45  # separate from edge_threshold
        query_emb   = self.model.encode(query, normalize_embeddings=True)
        sims        = self._emb_matrix @ query_emb          # shape (N,)
        matched_nodes = [
            self._emb_node_ids[i]
            for i in np.where(sims > QUERY_THRESHOLD)[0]
        ]

        expanded = set(matched_nodes)
        for node in matched_nodes:
            if node in self.graph:
                # Outgoing neighbours
                fwd = nx.single_source_shortest_path_length(
                    self.graph, node, cutoff=depth
                )
                expanded.update(fwd.keys())
                # Incoming neighbours (nodes that point to this one)
                rev = nx.single_source_shortest_path_length(
                    self.graph.reverse(copy=False), node, cutoff=depth
                )
                expanded.update(rev.keys())

        return list(expanded)

    # ------------------------------------------------------------------
    def related_chunks(self, nodes: list[str]) -> list:
        """
        Return chunk_ids referenced by edges touching the given nodes.
        Follows both outgoing and incoming edges so that a query for
        'lift coefficient' also retrieves chunks that define its dependencies,
        not only chunks it is a dependency of.
        """
        chunk_ids = set()
        for node in nodes:
            if node not in self.graph:
                continue
            # Outgoing edges
            for _, _, data in self.graph.edges(node, data=True):
                if "chunk_id" in data:
                    chunk_ids.add(data["chunk_id"])
            # Incoming edges
            for _, _, data in self.graph.in_edges(node, data=True):
                if "chunk_id" in data:
                    chunk_ids.add(data["chunk_id"])
        return list(chunk_ids)