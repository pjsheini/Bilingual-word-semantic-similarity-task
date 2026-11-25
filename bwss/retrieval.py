"""Batched nearest-neighbour search with NumPy.

Two scoring methods:

* ``nn``   - plain cosine similarity.
* ``csls`` - Cross-domain Similarity Local Scaling (Conneau et al., 2018):
  ``2*cos(x, y) - r_tgt(y) - r_src(x)``, where ``r`` is the mean cosine of a
  word to its k nearest neighbours in the other space. It penalises "hub"
  words that are close to everything and usually raises translation accuracy.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_BATCH = 512


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k largest scores per row, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def mean_knn_similarity(
    queries: np.ndarray, keys: np.ndarray, k: int = 10, batch: int = DEFAULT_BATCH
) -> np.ndarray:
    """For every query row, the mean cosine similarity to its k nearest keys."""
    out = np.empty(len(queries), dtype=np.float32)
    k = min(k, len(keys))
    for start in range(0, len(queries), batch):
        sims = queries[start:start + batch] @ keys.T
        part = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
        out[start:start + batch] = part.mean(axis=1)
    return out


def cached_mean_knn_similarity(
    queries: np.ndarray,
    keys: np.ndarray,
    k: int = 10,
    cache_dir: Optional[Path] = None,
    batch: int = DEFAULT_BATCH,
) -> np.ndarray:
    """``mean_knn_similarity`` with an on-disk cache keyed by the data itself.

    Computing r_tgt for a 200k vocabulary takes minutes, so it is reused
    across runs and across both CSLS methods of the same language pair.
    """
    if cache_dir is None:
        return mean_knn_similarity(queries, keys, k, batch)
    digest = hashlib.sha1()
    for arr in (queries, keys):
        digest.update(str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr[:: max(1, len(arr) // 2000)]).tobytes())
    digest.update(str(k).encode())
    path = Path(cache_dir) / f"csls-{digest.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path)
    result = mean_knn_similarity(queries, keys, k, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result


def search(
    queries: np.ndarray,
    targets: np.ndarray,
    k: int = 10,
    method: str = "nn",
    *,
    target_penalty: Optional[np.ndarray] = None,
    query_penalty: Optional[np.ndarray] = None,
    exclude: Optional[np.ndarray] = None,
    batch: int = DEFAULT_BATCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k target indices and scores for each query row.

    For ``method="csls"`` pass ``target_penalty`` (r_tgt for every target row)
    and ``query_penalty`` (r_src for every query row).
    ``exclude`` gives, per query, one target index to skip (or -1), used when
    source and target are the same shared space so a word can't find itself.
    """
    if method not in ("nn", "csls"):
        raise ValueError(f"unknown method {method!r}")
    if method == "csls" and (target_penalty is None or query_penalty is None):
        raise ValueError("csls needs target_penalty and query_penalty")
    n = len(queries)
    idx_out = np.empty((n, min(k, len(targets))), dtype=np.int64)
    score_out = np.empty(idx_out.shape, dtype=np.float32)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        scores = queries[start:stop] @ targets.T
        if method == "csls":
            scores *= 2
            scores -= target_penalty[None, :]
            scores -= query_penalty[start:stop, None]
        if exclude is not None:
            rows = np.arange(stop - start)
            cols = exclude[start:stop]
            mask = cols >= 0
            scores[rows[mask], cols[mask]] = -np.inf
        top = _top_k(scores, k)
        idx_out[start:stop] = top
        score_out[start:stop] = np.take_along_axis(scores, top, axis=1)
    return idx_out, score_out
