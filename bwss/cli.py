"""Command-line interface: ``python -m bwss <command> ...``.

Commands
--------
evaluate    P@1/5/10 for one dictionary
run-all     every dictionary in a folder, with a summary table
neighbours  nearest words to one or more words
analogy     a : b :: c : ?
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .embeddings import Embeddings
from .evaluate import EvalResult, evaluate, load_dictionary, parse_pair, write_predictions
from .explore import MultiSpace

KS = (1, 5, 10)


# --------------------------------------------------------------- helpers
def _load(path: str, args: argparse.Namespace, cache: Dict[str, Embeddings]) -> Embeddings:
    if path not in cache:
        t0 = time.time()
        emb = Embeddings.load(path, max_vocab=args.max_vocab, clean=args.clean, lowercase=args.lowercase)
        print(f"  loaded {path}: {len(emb):,} words x {emb.dim} dims ({time.time() - t0:.1f}s)", file=sys.stderr)
        cache[path] = emb
    return cache[path]


def _find_vectors(folder: Path, lang: str) -> Optional[Path]:
    for ext in (".npz", ".vec"):
        p = folder / f"{lang}{ext}"
        if p.exists():
            return p
    return None


def format_table(results: Sequence[EvalResult]) -> str:
    header = ["pair", "method", "evaluated", "coverage"] + [f"P@{k}" for k in KS] + ["time"]
    rows = [header]
    for r in results:
        rows.append(
            [r.pair, r.method, f"{r.evaluated_words}/{r.dictionary_words}", f"{r.coverage:.1f}%"]
            + [f"{r.precision[k]:.1f}" for k in KS]
            + [f"{r.seconds:.1f}s"]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_markdown(results: Sequence[EvalResult]) -> str:
    header = "| Pair | Method | Evaluated words | Coverage | P@1 | P@5 | P@10 |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    rows = [
        f"| {r.pair} | {r.method.upper()} | {r.evaluated_words:,} / {r.dictionary_words:,} | "
        f"{r.coverage:.1f}% | {r.precision[1]:.1f} | {r.precision[5]:.1f} | {r.precision[10]:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep] + rows)


def _print_examples(result: EvalResult, n: int) -> None:
    if n <= 0:
        return
    print(f"\nExamples ({result.pair}, {result.method}):")
    for p in result.predictions[:n]:
        mark = f"hit@{p.hit_rank}" if p.hit_rank else "miss"
        print(f"  {p.source:<16} gold={'|'.join(p.gold):<24} {mark:<7} -> {', '.join(p.predicted[:5])}")


def _methods(value: str) -> List[str]:
    return ["nn", "csls"] if value == "both" else [value]


def _write_outputs(results: List[EvalResult], out_dir: Optional[str]) -> None:
    if not out_dir:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        write_predictions(r, out / f"predictions-{r.pair}-{r.method}.tsv")
    (out / "summary.json").write_text(json.dumps([r.summary() for r in results], indent=2), encoding="utf-8")
    (out / "summary.md").write_text(format_markdown(results) + "\n", encoding="utf-8")
    print(f"\nWrote summary.md, summary.json and per-word predictions to {out}/", file=sys.stderr)


# --------------------------------------------------------------- commands
def cmd_evaluate(args: argparse.Namespace) -> int:
    cache: Dict[str, Embeddings] = {}
    if args.shared:
        src = tgt = _load(args.shared, args, cache)
    elif args.src and args.tgt:
        src, tgt = _load(args.src, args, cache), _load(args.tgt, args, cache)
    else:
        print("error: give --shared FILE, or both --src and --tgt", file=sys.stderr)
        return 2
    s, t = parse_pair(args.dict)
    results = [
        evaluate(src, tgt, load_dictionary(args.dict), KS, m, cache_dir=args.cache_dir, pair=f"{s}-{t}")
        for m in _methods(args.method)
    ]
    print(format_table(results))
    for r in results:
        _print_examples(r, args.examples)
    _write_outputs(results, args.out_dir)
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    cache: Dict[str, Embeddings] = {}
    results: List[EvalResult] = []
    for dict_path in sorted(Path(args.dicts_dir).glob("*-*.txt")):
        s, t = parse_pair(dict_path)
        if args.shared:
            src = tgt = _load(args.shared, args, cache)
        else:
            sp, tp = _find_vectors(Path(args.vectors_dir), s), _find_vectors(Path(args.vectors_dir), t)
            if not sp or not tp:
                print(f"  skipping {dict_path.name}: no {s}/{t} vectors in {args.vectors_dir}", file=sys.stderr)
                continue
            src, tgt = _load(str(sp), args, cache), _load(str(tp), args, cache)
        dictionary = load_dictionary(dict_path)
        for m in _methods(args.method):
            print(f"  evaluating {s}-{t} ({m})...", file=sys.stderr)
            results.append(evaluate(src, tgt, dictionary, KS, m, cache_dir=args.cache_dir, pair=f"{s}-{t}"))
    if not results:
        print("No dictionaries could be evaluated.", file=sys.stderr)
        return 1
    print()
    print(format_table(results))
    _write_outputs(results, args.out_dir)
    return 0


def _spaces(args: argparse.Namespace) -> MultiSpace:
    cache: Dict[str, Embeddings] = {}
    return MultiSpace([_load(p, args, cache) for p in args.vec])


def cmd_neighbours(args: argparse.Namespace) -> int:
    space = _spaces(args)
    for word in args.words:
        hits = space.neighbours(word, args.n)
        if hits is None:
            print(f"{word}: not in vocabulary")
        else:
            print(f"{word}: " + ", ".join(str(h) for h in hits))
    return 0


def cmd_analogy(args: argparse.Namespace) -> int:
    space = _spaces(args)
    triples = [args.words[i:i + 3] for i in range(0, len(args.words), 3)]
    for triple in triples:
        if len(triple) != 3:
            print(f"ignoring incomplete analogy {triple}")
            continue
        a, b, c = triple
        hits = space.analogy(a, b, c, args.n)
        if hits is None:
            print(f"{a} : {b} :: {c} : ?   (unknown: {', '.join(space.missing(triple))})")
        else:
            print(f"{a} : {b} :: {c} : {hits[0].word}   [" + ", ".join(str(h) for h in hits) + "]")
    return 0


# --------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bwss", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def load_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--max-vocab", type=int, default=200_000,
                        help="keep the N most frequent words of each file (default 200000; 0 = all)")
        sp.add_argument("--clean", action="store_true",
                        help="drop 1-char/punctuation tokens and near-duplicates (original project's filter)")
        sp.add_argument("--lowercase", action="store_true", help="lowercase all words")

    def eval_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--method", choices=["nn", "csls", "both"], default="both",
                        help="retrieval: cosine nearest neighbour, CSLS, or both (default)")
        sp.add_argument("--out-dir", default="results", help="where to write summaries ('' to skip)")
        sp.add_argument("--cache-dir", default=".cache", help="cache for CSLS statistics")
        sp.add_argument("--shared", help="one embedding file holding both languages (e.g. the BUC model)")
        load_opts(sp)

    e = sub.add_parser("evaluate", help="evaluate one dictionary")
    e.add_argument("--src", help="source-language embeddings (.vec or .npz)")
    e.add_argument("--tgt", help="target-language embeddings, aligned with --src")
    e.add_argument("--dict", required=True, help="dictionary file, e.g. dicts/en-es.5000-6500.txt")
    e.add_argument("--examples", type=int, default=10, help="print N example translations")
    eval_opts(e)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("run-all", help="evaluate every dictionary in a folder")
    r.add_argument("--vectors-dir", default="vectors", help="folder with <lang>.npz or <lang>.vec files")
    r.add_argument("--dicts-dir", default="dicts")
    eval_opts(r)
    r.set_defaults(func=cmd_run_all)

    for name, func, help_ in (("neighbours", cmd_neighbours, "nearest words"),
                              ("analogy", cmd_analogy, "a : b :: c : ?  (give words in threes)")):
        x = sub.add_parser(name, help=help_)
        x.add_argument("--vec", action="append", required=True,
                       help="embedding file; repeat for several aligned languages")
        x.add_argument("-n", type=int, default=10, help="results per query")
        x.add_argument("words", nargs="+", help="words; use lang:word to pick a language, e.g. es:no")
        load_opts(x)
        x.set_defaults(func=func)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_vocab", None) == 0:
        args.max_vocab = None
    return args.func(args)
