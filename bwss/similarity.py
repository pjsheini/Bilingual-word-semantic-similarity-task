"""Word-similarity evaluation: Spearman correlation with human judgements.

Each dataset is a list of word pairs with a human similarity or relatedness
score. For every pair whose two words are in the vocabulary, the cosine
similarity of their vectors is computed; the result is Spearman's rank
correlation between those cosines and the human scores (higher is better,
1.0 = same ranking as people).

Monolingual datasets (e.g. German GUR350) test whether meaning is kept
*within* a language; cross-lingual ones (e.g. English-German RG65, where
word 1 is English and word 2 German) test the shared bilingual space.

This is the task of the original ``WE-Word-similarity-task`` repository.
"""

import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .embeddings import Embeddings

PathLike = Union[str, Path]
Pair = Tuple[str, str, float]

_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def load_pairs(path: PathLike) -> List[Pair]:
    """Read ``word1 <sep> word2 <sep> score`` lines.

    The separator is detected per file: tab, ``:`` (the GUR/ZG format used by
    the original script), ``,`` or spaces. Comment lines (``#``) and header
    lines whose third field isn't a number are skipped.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return []
    sample = lines[min(1, len(lines) - 1)]
    sep: Optional[str]
    if "\t" in sample:
        sep = "\t"
    elif sample.count(":") >= 2:
        sep = ":"
    elif "," in sample:
        sep = ","
    else:
        sep = None  # any whitespace
    pairs: List[Pair] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(sep)]
        if len(parts) < 3 or not _NUMBER.match(parts[2]):
            continue
        pairs.append((parts[0], parts[1], float(parts[2])))
    return pairs


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get the average rank (like scipy's default)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


class Lookup:
    """Finds a dataset word in a vocabulary: exact, then lowercase, then
    (optionally) ignoring accents, picking the most frequent matching word.

    Accent folding can help with datasets distributed without accents, such
    as the Spanish WS353 (``quimica`` for ``química``), when the vocabulary
    has no accent-less spellings. Web-trained vocabularies usually do, so it
    is off by default.
    """

    def __init__(self, emb: Embeddings, fold_accents: bool = False) -> None:
        self.emb = emb
        self.fold_accents = fold_accents
        self._folded: Dict[str, int] = {}
        if fold_accents:
            for i, w in enumerate(emb.words):  # frequency order: first wins
                self._folded.setdefault(strip_accents(w.lower()), i)

    def __call__(self, word: str) -> Optional[int]:
        idx = self.emb.index
        word = word.strip()
        if self.fold_accents:
            # Accent-less datasets: prefer the most frequent accented form
            # over a rare unaccented misspelling that happens to be in vocab.
            i = self._folded.get(strip_accents(word.lower()))
            if i is not None:
                return i
        for candidate in (word, word.lower()):
            if candidate in idx:
                return idx[candidate]
        return None


@dataclass
class SimilarityResult:
    dataset: str
    langs: str
    pairs: int
    found: int
    spearman: float
    seconds: float

    @property
    def coverage(self) -> float:
        return 100.0 * self.found / max(1, self.pairs)

    def summary(self) -> dict:
        d = asdict(self)
        d["spearman"] = round(self.spearman, 4)
        d["coverage"] = round(self.coverage, 2)
        d["seconds"] = round(self.seconds, 2)
        return d


def evaluate_similarity(
    pairs: Sequence[Pair],
    left: Embeddings,
    right: Optional[Embeddings] = None,
    fold_accents: bool = False,
    dataset: str = "",
    langs: str = "",
) -> SimilarityResult:
    """Spearman correlation between cosine similarity and human scores.

    ``left`` holds the first word of each pair, ``right`` the second (defaults
    to ``left``; give a different, aligned space for cross-lingual datasets).
    Pairs with an out-of-vocabulary word are skipped and reported as coverage.
    """
    t0 = time.time()
    right = right or left
    find_l = Lookup(left, fold_accents)
    find_r = find_l if right is left else Lookup(right, fold_accents)
    cosines: List[float] = []
    gold: List[float] = []
    for w1, w2, score in pairs:
        i, j = find_l(w1), find_r(w2)
        if i is None or j is None:
            continue
        cosines.append(float(left.vectors[i] @ right.vectors[j]))
        gold.append(score)
    return SimilarityResult(
        dataset=dataset,
        langs=langs,
        pairs=len(pairs),
        found=len(cosines),
        spearman=spearman(cosines, gold),
        seconds=time.time() - t0,
    )


def dataset_langs(path: PathLike) -> Tuple[str, str]:
    """Languages from the parent folder: ``de/gur350.txt`` -> ("de", "de"),
    ``en-de/rg65.txt`` -> ("en", "de")."""
    folder = Path(path).parent.name
    a, _, b = folder.partition("-")
    return a, (b or a)
