"""Loading, cleaning and saving word embeddings.

Supports two formats:

* ``.vec`` - fastText / MUSE text format: an optional ``<count> <dim>`` header,
  then one ``word v1 v2 ... vd`` line per word, most frequent words first.
* ``.npz`` - NumPy archive with a ``words`` array and a ``vectors`` matrix.
  Much faster to load; ``Embeddings.save`` writes it.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

PathLike = Union[str, Path]

# A "valid" word starts and ends with an alphanumeric character, so tokens
# like "-ab", "ab_", "a." are dropped (same rule as the original load.py).
_VALID_WORD = re.compile(r"^[^\W_].*[^\W_]$")
# Characters ignored when spotting near-duplicates ("U.S.", "U.S", "US_" == "US").
_IGNORED_CHARS = re.compile(r"[\W_]")


def is_valid_word(word: str) -> bool:
    return len(word) > 1 and _VALID_WORD.match(word) is not None


def canonical_form(word: str) -> str:
    return _IGNORED_CHARS.sub("", word)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 length (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embeddings:
    """A vocabulary plus a row-normalised ``float32`` matrix (one row per word).

    Rows are unit length, so a dot product is a cosine similarity.
    """

    def __init__(self, words: List[str], vectors: np.ndarray, name: str = "") -> None:
        if len(words) != vectors.shape[0]:
            raise ValueError(f"{len(words)} words but {vectors.shape[0]} vectors")
        self.name = name
        self.words: List[str] = list(words)
        self.vectors: np.ndarray = normalize_rows(np.asarray(vectors, dtype=np.float32))
        # Keep the first (most frequent) row when a word appears twice.
        self.index: Dict[str, int] = {}
        for i, w in enumerate(self.words):
            self.index.setdefault(w, i)

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def vector(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(
        cls,
        path: PathLike,
        max_vocab: Optional[int] = None,
        clean: bool = False,
        lowercase: bool = False,
    ) -> "Embeddings":
        """Load ``.vec`` or ``.npz`` embeddings.

        max_vocab: keep only the first N words (files are frequency-sorted).
        clean:     drop one-character / punctuation-edged tokens and
                   near-duplicates, as the original project did.
        lowercase: lowercase words, keeping the first occurrence.
        """
        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            words = [str(w) for w in data["words"]]
            vectors = data["vectors"]
        else:
            words, vectors = _read_vec(path, None if (clean or lowercase) else max_vocab)
        name = path.stem
        if lowercase:
            words = [w.lower() for w in words]
        keep = _select(words, clean=clean, dedupe=lowercase)
        if max_vocab is not None:
            keep = keep[:max_vocab]
        return cls([words[i] for i in keep], vectors[keep], name=name)

    def save(self, path: PathLike) -> None:
        """Save as ``.npz`` (fast) or ``.vec`` (text, interoperable)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(path, words=np.array(self.words), vectors=self.vectors)
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{len(self)} {self.dim}\n")
            for word, row in zip(self.words, self.vectors):
                f.write(word + " " + " ".join(f"{x:.5f}" for x in row) + "\n")


def _select(words: List[str], clean: bool, dedupe: bool) -> List[int]:
    """Indices of the rows to keep, in original order."""
    keep: List[int] = []
    seen = set()
    for i, w in enumerate(words):
        if clean and not is_valid_word(w):
            continue
        key = canonical_form(w) if clean else w
        if (clean or dedupe) and key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def _read_vec(path: Path, limit: Optional[int]):
    words: List[str] = []
    rows: List[np.ndarray] = []
    dim: Optional[int] = None
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n").split(" ")
        if len(first) == 2 and all(t.isdigit() for t in first):
            dim = int(first[1])  # header line
        else:
            f.seek(0)
        for line in f:
            if limit is not None and len(words) >= limit:
                break
            parts = line.rstrip("\n ").split(" ")
            if dim is None:
                dim = len(parts) - 1
            if len(parts) < dim + 1:
                continue  # malformed line (e.g. missing word)
            # Words may contain spaces: everything before the last `dim` fields.
            words.append(" ".join(parts[:-dim]))
            rows.append(np.asarray(parts[-dim:], dtype=np.float32))
    if not rows:
        raise ValueError(f"No vectors found in {path}")
    return words, np.vstack(rows)


def load_many(paths: Iterable[PathLike], **kwargs) -> List[Embeddings]:
    return [Embeddings.load(p, **kwargs) for p in paths]
