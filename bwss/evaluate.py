"""Word-translation evaluation (precision@k) against a bilingual dictionary."""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .embeddings import Embeddings
from .retrieval import cached_mean_knn_similarity, search

PathLike = Union[str, Path]


def load_dictionary(path: PathLike) -> Dict[str, List[str]]:
    """Read a MUSE-style ``source target`` file into {source: [targets]}.

    A source word may have several valid translations (several lines).
    Accepts space- or tab-separated files.
    """
    pairs: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0], parts[1]
            targets = pairs.setdefault(src, [])
            if tgt not in targets:
                targets.append(tgt)
    return pairs


@dataclass
class Prediction:
    source: str
    gold: List[str]
    predicted: List[str]
    scores: List[float]
    hit_rank: Optional[int]  # 1-based rank of the first correct translation


@dataclass
class EvalResult:
    pair: str
    method: str
    dictionary_words: int  # distinct source words in the dictionary
    evaluated_words: int  # source and at least one gold target in vocabulary
    precision: Dict[int, float]  # k -> P@k over evaluated words (in %)
    seconds: float
    predictions: List[Prediction] = field(default_factory=list, repr=False)

    @property
    def coverage(self) -> float:
        return 100.0 * self.evaluated_words / max(1, self.dictionary_words)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("predictions")
        d["coverage"] = round(self.coverage, 2)
        d["precision"] = {f"P@{k}": round(v, 2) for k, v in self.precision.items()}
        d["seconds"] = round(self.seconds, 1)
        return d


def evaluate(
    src: Embeddings,
    tgt: Embeddings,
    dictionary: Dict[str, List[str]],
    ks: Sequence[int] = (1, 5, 10),
    method: str = "nn",
    csls_k: int = 10,
    cache_dir: Optional[PathLike] = None,
    pair: str = "",
) -> EvalResult:
    """Translate each dictionary source word by nearest-neighbour search.

    A word counts as correct at k if any gold translation is among the k best
    candidates. Words whose source form, or all of whose gold translations,
    are missing from the vocabularies are skipped (standard MUSE protocol);
    ``coverage`` reports how many were kept.

    If ``src`` and ``tgt`` are the same object (one shared bilingual space,
    like the original BUC model), the query word is excluded from its own
    candidates, and an identical-spelling gold translation counts as rank 1.
    """
    t0 = time.time()
    shared = src is tgt
    max_k = max(ks)

    queries: List[str] = []
    golds: List[List[str]] = []
    for word, targets in dictionary.items():
        in_vocab = [t for t in targets if t in tgt]
        if word in src and in_vocab:
            queries.append(word)
            golds.append(in_vocab)

    q_idx = np.array([src.index[w] for w in queries], dtype=np.int64)
    q_vecs = src.vectors[q_idx] if len(q_idx) else np.zeros((0, src.dim), np.float32)

    kwargs = {}
    if method == "csls":
        cache = Path(cache_dir) if cache_dir else None
        kwargs["target_penalty"] = cached_mean_knn_similarity(tgt.vectors, src.vectors, csls_k, cache)
        kwargs["query_penalty"] = cached_mean_knn_similarity(q_vecs, tgt.vectors, csls_k, None)
    if shared:
        kwargs["exclude"] = q_idx

    if len(queries):
        top_idx, top_scores = search(q_vecs, tgt.vectors, max_k, method, **kwargs)
    else:
        top_idx = np.zeros((0, max_k), np.int64)
        top_scores = np.zeros((0, max_k), np.float32)

    predictions: List[Prediction] = []
    hits = {k: 0 for k in ks}
    for word, gold, row, scores in zip(queries, golds, top_idx, top_scores):
        predicted = [tgt.words[i] for i in row]
        if shared and word in gold:
            predicted = [word] + predicted[:-1]
            scores = np.concatenate([[1.0], scores[:-1]])
        gold_set = set(gold)
        rank = next((r + 1 for r, p in enumerate(predicted) if p in gold_set), None)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        predictions.append(Prediction(word, gold, predicted, [float(s) for s in scores], rank))

    n = max(1, len(queries))
    return EvalResult(
        pair=pair,
        method=method,
        dictionary_words=len(dictionary),
        evaluated_words=len(queries),
        precision={k: 100.0 * hits[k] / n for k in ks},
        seconds=time.time() - t0,
        predictions=predictions,
    )


def write_predictions(result: EvalResult, path: PathLike, top: int = 10) -> None:
    """One TSV row per evaluated word: source, gold, hit rank, top predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source\tgold\thit_rank\tpredictions\n")
        for p in result.predictions:
            preds = ", ".join(f"{w} ({s:.3f})" for w, s in zip(p.predicted[:top], p.scores[:top]))
            f.write(f"{p.source}\t{'|'.join(p.gold)}\t{p.hit_rank or ''}\t{preds}\n")


def parse_pair(dict_path: PathLike) -> Tuple[str, str]:
    """``en-es.5000-6500.txt`` -> ("en", "es")."""
    stem = Path(dict_path).name.split(".")[0]
    src, _, tgt = stem.partition("-")
    if not src or not tgt:
        raise ValueError(f"Can't read a language pair from {dict_path}")
    return src, tgt
