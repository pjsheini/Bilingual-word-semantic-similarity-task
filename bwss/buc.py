"""Bilingual / multilingual Universal Concepts (BUC / MUC) word embeddings.

Implements the method of Sheinidashtegol & Musaev (2019), "Learning
Cross-lingual Word Embeddings with Universal Concepts", IJWSC 10(3), 13-20:

1. **Concepts.** Build an undirected graph from bilingual dictionaries
   (words are nodes, translation pairs are edges). Each connected component
   is one *universal concept* with its own ID.
2. **Rewrite.** In each monolingual corpus, replace every word that belongs
   to a concept with the concept ID, then train on all corpora together, so
   ``fox`` / ``zorro`` / ``fuchs`` share contexts across languages.
3. **Train** ordinary word2vec or fastText on the rewritten corpus
   (the paper uses a context window of 10).
4. **Expand.** Give every word in a concept the concept's vector and write
   one aligned ``<lang>.npz`` file per language (or one shared file), ready
   for ``python -m bwss run-all`` and ``python -m bwss similarity``.

Differences from the original scripts, all on purpose:

* Words are kept per language (``en:no`` and ``es:no`` are different nodes)
  unless a dictionary pair links them, so identical spellings don't merge
  unrelated words.
* Test words can be excluded from the graph (``exclude``), so concepts never
  contain a test pair. Use the MUSE ``0-5000`` training dictionaries, or the
  full ones with the ``5000-6500`` test files excluded.
* The corpus is rewritten line by line (the original read 1 KB chunks, which
  split words at chunk boundaries).
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .embeddings import Embeddings
from .evaluate import load_dictionary, parse_pair

PathLike = Union[str, Path]
Node = Tuple[str, str]  # (lang, word)

CONCEPT_PREFIX = "UC_"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ concepts
class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Node, Node] = {}

    def find(self, x: Node) -> Node:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Node, b: Node) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class Concepts:
    """Universal concepts: concept ID -> {lang: [words]}, plus the reverse map."""

    members: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lookup: Dict[Node, str] = {}
        for cid, langs in self.members.items():
            for lang, words in langs.items():
                for w in words:
                    self.lookup[(lang, w)] = cid

    def __len__(self) -> int:
        return len(self.members)

    def concept_of(self, lang: str, word: str) -> Optional[str]:
        return self.lookup.get((lang, word))

    def sizes(self) -> Counter:
        return Counter(sum(len(w) for w in langs.values()) for langs in self.members.values())

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for cid, langs in self.members.items():
                f.write(json.dumps({"id": cid, "words": langs}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: PathLike, legacy_langs: Sequence[str] = ()) -> "Concepts":
        """Read ``Concepts.save`` output, or the original project's files.

        Original files are JSON lines with ``UC_Nodelist`` or ``UC`` and
        ``GUID`` and no language information, so their words are assigned to
        every language in ``legacy_langs``.
        """
        members: Dict[str, Dict[str, List[str]]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "words" in d:
                    members[d["id"]] = d["words"]
                    continue
                words = (d.get("UC_Nodelist") or d.get("UC"))[0]
                cid = CONCEPT_PREFIX + d["GUID"][0]
                if not legacy_langs:
                    raise ValueError("legacy concept file: pass legacy_langs, e.g. ['en', 'es']")
                members[cid] = {lang: list(words) for lang in legacy_langs}
        return cls(members)


def dictionary_edges(path: PathLike, lowercase: bool = True) -> Iterator[Tuple[Node, Node]]:
    src, tgt = parse_pair(path)
    for s, targets in load_dictionary(path).items():
        for t in targets:
            if lowercase:
                s, t = s.lower(), t.lower()
            yield (src, s), (tgt, t)


def build_concepts(
    dictionaries: Iterable[PathLike],
    exclude: Iterable[PathLike] = (),
    max_size: Optional[int] = None,
    lowercase: bool = True,
) -> Tuple[Concepts, dict]:
    """Connected components of the translation graph, one concept each.

    ``exclude``: dictionaries (e.g. the MUSE test sets) whose words, in their
    language, must not join any concept. ``max_size``: drop components with
    more words than this (very large components are chains of loosely
    related words); ``None`` keeps all, as the paper does.
    Returns the concepts and a stats dict.
    """
    banned: Set[Node] = set()
    for path in exclude:
        for a, b in dictionary_edges(path, lowercase):
            banned.update((a, b))

    uf = _UnionFind()
    edges = kept = 0
    for path in dictionaries:
        for a, b in dictionary_edges(path, lowercase):
            edges += 1
            if a in banned or b in banned:
                continue
            uf.union(a, b)
            kept += 1

    groups: Dict[Node, List[Node]] = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)

    members: Dict[str, Dict[str, List[str]]] = {}
    too_big = 0
    for nodes in sorted(groups.values(), key=lambda ns: (-len(ns), min(ns))):
        if max_size is not None and len(nodes) > max_size:
            too_big += 1
            continue
        langs: Dict[str, List[str]] = {}
        for lang, word in sorted(nodes):
            langs.setdefault(lang, []).append(word)
        members[f"{CONCEPT_PREFIX}{len(members):06d}"] = langs

    concepts = Concepts(members)
    sizes = concepts.sizes()
    stats = {
        "dictionary_pairs": edges,
        "pairs_used": kept,
        "pairs_excluded": edges - kept,
        "banned_words": len(banned),
        "concepts": len(concepts),
        "words_in_concepts": len(concepts.lookup),
        "dropped_large_concepts": too_big,
        "largest_concept": max(sizes) if sizes else 0,
        "mean_concept_size": round(sum(k * v for k, v in sizes.items()) / max(1, len(concepts)), 2),
    }
    return concepts, stats


# ------------------------------------------------------------------- rewrite
def rewrite_corpus(
    concepts: Concepts,
    lang: str,
    src: PathLike,
    dst: PathLike,
    lowercase: bool = True,
) -> dict:
    """Replace concept words with concept IDs; prefix other tokens ``lang:``.

    The prefix keeps words of different languages apart in the shared model
    (Spanish ``no`` and English ``no`` only meet if a concept joins them).
    Input is whitespace-tokenised text, one sentence or paragraph per line.
    """
    tokens = replaced = 0
    lookup = concepts.lookup
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    prefix = lang + ":"
    with open(src, encoding="utf-8", errors="replace") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            words = line.lower().split() if lowercase else line.split()
            if not words:
                continue
            out = []
            for w in words:
                cid = lookup.get((lang, w))
                if cid is None:
                    out.append(prefix + w)
                else:
                    out.append(cid)
                    replaced += 1
            tokens += len(words)
            fout.write(" ".join(out) + "\n")
    return {"lang": lang, "tokens": tokens, "replaced": replaced,
            "replaced_pct": round(100.0 * replaced / max(1, tokens), 2)}


# --------------------------------------------------------------------- train
class _Sentences:
    """Restartable iterator over the lines of several files (for gensim)."""

    def __init__(self, paths: Sequence[PathLike]) -> None:
        self.paths = list(paths)

    def __iter__(self) -> Iterator[List[str]]:
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    toks = line.split()
                    if toks:
                        yield toks


def train(
    corpora: Sequence[PathLike],
    algorithm: str = "word2vec",
    dim: int = 300,
    window: int = 10,
    min_count: int = 5,
    epochs: int = 5,
    sg: bool = True,
    workers: int = 4,
    seed: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Train on the rewritten corpora; returns (tokens by frequency, vectors).

    Needs gensim (``pip install gensim``). fastText is used with subword
    n-grams; concept IDs are atomic tokens either way.
    """
    try:
        from gensim.models import FastText, Word2Vec
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Training needs gensim: pip install gensim") from e
    cls = {"word2vec": Word2Vec, "fasttext": FastText}[algorithm]
    model = cls(sentences=_Sentences(corpora), vector_size=dim, window=window, min_count=min_count,
                epochs=epochs, sg=int(sg), workers=workers, seed=seed)
    tokens = list(model.wv.index_to_key)  # most frequent first
    return tokens, np.asarray(model.wv.vectors, dtype=np.float32)


# -------------------------------------------------------------------- expand
def expand(
    tokens: Sequence[str],
    vectors: np.ndarray,
    concepts: Concepts,
    langs: Sequence[str],
) -> Dict[str, Embeddings]:
    """Per-language aligned spaces: concept IDs become their member words.

    Words keep the frequency order of their token. A word that is both a
    concept member and (rarely) a plain token keeps the more frequent one.
    """
    out: Dict[str, Tuple[List[str], List[int]]] = {lang: ([], []) for lang in langs}
    seen: Dict[str, Set[str]] = {lang: set() for lang in langs}
    for row, tok in enumerate(tokens):
        if tok.startswith(CONCEPT_PREFIX) and tok in concepts.members:
            pairs = [(lang, w) for lang, ws in concepts.members[tok].items() for w in ws]
        elif ":" in tok:
            lang, _, w = tok.partition(":")
            pairs = [(lang, w)]
        else:
            continue
        for lang, w in pairs:
            if lang in out and w not in seen[lang]:
                seen[lang].add(w)
                out[lang][0].append(w)
                out[lang][1].append(row)
    return {lang: Embeddings(words, vectors[rows] if rows else np.zeros((0, vectors.shape[1])), lang)
            for lang, (words, rows) in out.items()}


def shared_space(spaces: Dict[str, Embeddings]) -> Embeddings:
    """One file with every language's words, like the original BUC ``.vec``
    (the first language wins when a spelling occurs in several)."""
    words: List[str] = []
    rows: List[np.ndarray] = []
    seen: Set[str] = set()
    for emb in spaces.values():
        for w, v in zip(emb.words, emb.vectors):
            if w not in seen:
                seen.add(w)
                words.append(w)
                rows.append(v)
    dim = next(iter(spaces.values())).dim
    return Embeddings(words, np.vstack(rows) if rows else np.zeros((0, dim)), "shared")
