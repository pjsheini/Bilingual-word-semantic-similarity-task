"""Build demo cross-lingual embeddings to run the evaluation on.

The original BUC model files are not available, so this script builds a
transparent baseline from public resources:

1. Monolingual 300-d vectors from spaCy's ``*_lg`` models (en, es, de, ru),
   with words ranked by frequency using the ``wordfreq`` package.
2. Each non-English space is mapped onto English with an orthogonal
   (Procrustes) transform, learned *without* a bilingual dictionary:
   a seed of identically spelled words and numerals, refined by
   self-learning with CSLS (in the style of Artetxe et al., 2018).
   Any word that appears in a test dictionary is kept out of the seed.

Output: ``vectors/<lang>.npz``, ready for ``python -m bwss run-all``.

Requirements (only for this script): ``pip install spacy wordfreq``, plus the
spaCy model wheels, which you can unzip instead of installing:

    curl -LO https://github.com/explosion/spacy-models/releases/download/\\
en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
    python -m zipfile -e en_core_web_lg-3.8.0-py3-none-any.whl models/

    python tools/build_demo_vectors.py --models-dir models --out vectors
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bwss.embeddings import Embeddings, normalize_rows  # noqa: E402
from bwss.evaluate import load_dictionary  # noqa: E402

MODEL_NAMES = {"en": "en_core_web_lg", "es": "es_core_news_lg",
               "de": "de_core_news_lg", "ru": "ru_core_news_lg"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- export
def export_language(models_dir: Path, lang: str, max_vocab: int) -> Tuple[List[str], np.ndarray]:
    """Frequency-ranked, lowercase vocabulary with spaCy vectors."""
    from spacy.strings import StringStore
    from spacy.vectors import Vectors
    from wordfreq import top_n_list

    name = MODEL_NAMES[lang]
    vocab_dirs = list((models_dir / name).glob(f"{name}-*/vocab"))
    if not vocab_dirs:
        raise SystemExit(f"Can't find {name}/{name}-*/vocab under {models_dir}")
    vectors = Vectors().from_disk(vocab_dirs[0])
    strings = StringStore().from_disk(vocab_dirs[0] / "strings.json")
    table = vectors.data if isinstance(vectors.data, np.ndarray) else np.asarray(vectors.data)

    words: List[str] = []
    rows: List[int] = []
    seen: Set[str] = set()
    # wordfreq gives a frequency ranking; spaCy supplies the vectors.
    candidates = top_n_list(lang, int(max_vocab * 2), wordlist="large")
    candidates += [str(i) for i in range(0, 2101)]  # numerals, used as anchors
    for w in candidates:
        if w in seen:
            continue
        row = vectors.key2row.get(strings[w], -1)
        if row < 0:
            continue
        seen.add(w)
        words.append(w)
        rows.append(row)
        if len(words) >= max_vocab:
            break
    return words, np.asarray(table[rows], dtype=np.float32)


def normalize(m: np.ndarray) -> np.ndarray:
    """Unit length, mean-centre, unit length again (standard for mapping)."""
    m = normalize_rows(m)
    m = m - m.mean(axis=0, keepdims=True)
    return normalize_rows(m).astype(np.float32)


# ---------------------------------------------------------------- mapping
def procrustes(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x.T @ z)
    return (u @ vt).astype(np.float32)


def knn_mean(a: np.ndarray, b: np.ndarray, k: int = 10, batch: int = 2048) -> np.ndarray:
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), batch):
        sims = a[s:s + batch] @ b.T
        out[s:s + batch] = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
    return out


def csls_argmax(a: np.ndarray, b: np.ndarray, ra: np.ndarray, rb: np.ndarray, batch: int = 2048) -> np.ndarray:
    out = np.empty(len(a), np.int64)
    for s in range(0, len(a), batch):
        scores = 2 * (a[s:s + batch] @ b.T) - rb[None, :] - ra[s:s + batch, None]
        out[s:s + batch] = scores.argmax(axis=1)
    return out


def self_learning(x: np.ndarray, z: np.ndarray, seed: List[Tuple[int, int]],
                  search_vocab: int, iterations: int) -> np.ndarray:
    """Alternate Procrustes fitting and dictionary induction (fwd+bwd CSLS)."""
    src_idx = np.array([s for s, _ in seed])
    tgt_idx = np.array([t for _, t in seed])
    xs, zs = x[:search_vocab], z[:search_vocab]
    for it in range(iterations):
        w = procrustes(x[src_idx], z[tgt_idx])
        xw = xs @ w
        rx, rz = knn_mean(xw, zs), knn_mean(zs, xw)
        fwd = csls_argmax(xw, zs, rx, rz)
        bwd = csls_argmax(zs, xw, rz, rx)
        pairs = set(zip(range(len(xs)), fwd.tolist())) | set(zip(bwd.tolist(), range(len(zs))))
        mutual = int(np.sum(bwd[fwd] == np.arange(len(xs))))
        src_idx = np.array([p[0] for p in pairs])
        tgt_idx = np.array([p[1] for p in pairs])
        log(f"    iteration {it + 1}/{iterations}: {len(pairs):,} pairs, {mutual:,} mutual")
    return procrustes(x[src_idx], z[tgt_idx])


def held_out_words(dicts_dir: Path, lang: str) -> Tuple[Set[str], Set[str]]:
    """Every word in the en<->lang test dictionaries (excluded from the seed)."""
    en_words: Set[str] = set()
    other: Set[str] = set()
    for path in (dicts_dir / f"en-{lang}.5000-6500.txt", dicts_dir / f"{lang}-en.5000-6500.txt"):
        if not path.exists():
            continue
        for s, targets in load_dictionary(path).items():
            if path.name.startswith("en-"):
                en_words.add(s)
                other.update(targets)
            else:
                other.add(s)
                en_words.update(targets)
    return en_words, other


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True, help="folder with unzipped spaCy *_lg model wheels")
    p.add_argument("--out", default="vectors")
    p.add_argument("--dicts-dir", default="dicts")
    p.add_argument("--langs", nargs="+", default=["es", "de", "ru"], help="languages to map onto English")
    p.add_argument("--max-vocab", type=int, default=200_000)
    p.add_argument("--seed-vocab", type=int, default=20_000, help="vocab used for the identical-string seed")
    p.add_argument("--search-vocab", type=int, default=20_000, help="vocab used during self-learning")
    p.add_argument("--iterations", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out)
    models = Path(args.models_dir)
    log("exporting en")
    en_words, en_raw = export_language(models, "en", args.max_vocab)
    z = normalize(en_raw)
    Embeddings(en_words, z, "en").save(out / "en.npz")
    en_index: Dict[str, int] = {w: i for i, w in enumerate(en_words)}

    for lang in args.langs:
        log(f"exporting {lang}")
        words, raw = export_language(models, lang, args.max_vocab)
        x = normalize(raw)
        held_en, held_xx = held_out_words(Path(args.dicts_dir), lang)
        seed = [
            (i, en_index[w])
            for i, w in enumerate(words[: args.seed_vocab])
            if w in en_index and en_index[w] < args.seed_vocab
            and w not in held_en and w not in held_xx
        ]
        log(f"  {lang}: {len(words):,} words, seed of {len(seed):,} identical strings/numerals")
        w = self_learning(x, z, seed, args.search_vocab, args.iterations)
        Embeddings(words, x @ w, lang).save(out / f"{lang}.npz")
        log(f"  wrote {out / (lang + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
