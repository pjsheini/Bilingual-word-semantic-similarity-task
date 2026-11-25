"""Nearest neighbours and analogies across one or more aligned spaces.

Several aligned spaces (e.g. en.npz and es.npz mapped into one space) are
searched together, so ``hombre : rey :: woman : ?`` works across languages.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import Embeddings, normalize_rows


@dataclass
class Hit:
    word: str
    lang: str
    score: float

    def __str__(self) -> str:
        return f"{self.word} [{self.lang}] {self.score:.3f}"


class MultiSpace:
    """Several aligned embedding spaces searched as one."""

    def __init__(self, spaces: Sequence[Embeddings]) -> None:
        if not spaces:
            raise ValueError("need at least one embedding space")
        self.spaces = list(spaces)

    def find(self, word: str, lang: Optional[str] = None) -> Optional[Tuple[Embeddings, np.ndarray]]:
        """Look a word up in ``lang``, or in the space where it is most frequent.

        Vocabularies overlap ("mujer" is also a rare English token), so
        without a language the word is taken from the space where it ranks
        highest. ``lang:word`` picks the language explicitly, e.g. ``es:no``.
        """
        if ":" in word and lang is None:
            maybe_lang, _, rest = word.partition(":")
            if any(s.name == maybe_lang for s in self.spaces):
                lang, word = maybe_lang, rest
        candidates = [s for s in self.spaces if (lang is None or s.name == lang) and word in s]
        if not candidates:
            return None
        space = min(candidates, key=lambda s: s.index[word])
        return space, space.vector(word)

    def nearest(self, vector: np.ndarray, n: int = 10, exclude: Sequence[str] = ()) -> List[Hit]:
        vector = normalize_rows(vector[None, :].astype(np.float32))[0]
        skip = {w.lower() for w in exclude}
        hits: List[Hit] = []
        for space in self.spaces:
            sims = space.vectors @ vector
            take = min(len(sims), n + len(skip) + 1)
            top = np.argpartition(sims, len(sims) - take)[-take:]
            hits.extend(Hit(space.words[i], space.name, float(sims[i])) for i in top)
        hits = [h for h in hits if h.word.lower() not in skip]
        return sorted(hits, key=lambda h: -h.score)[:n]

    def neighbours(self, word: str, n: int = 10) -> Optional[List[Hit]]:
        found = self.find(word)
        if found is None:
            return None
        return self.nearest(found[1], n, exclude=[word.split(":")[-1]])

    def analogy(self, a: str, b: str, c: str, n: int = 5) -> Optional[List[Hit]]:
        """``a : b :: c : ?``  ->  nearest words to b - a + c."""
        vecs = [self.find(w) for w in (a, b, c)]
        if any(v is None for v in vecs):
            return None
        va, vb, vc = (v[1] for v in vecs)
        words = [w.split(":")[-1] for w in (a, b, c)]
        return self.nearest(vb - va + vc, n, exclude=words)

    def missing(self, words: Sequence[str]) -> List[str]:
        return [w for w in words if self.find(w) is None]
