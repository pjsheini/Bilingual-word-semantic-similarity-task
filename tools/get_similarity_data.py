"""Download word-similarity benchmarks into data/similarity/ as plain TSV.

Every file is fetched from a pinned commit of a public GitHub repository and
rewritten as ``word1<TAB>word2<TAB>score``. The folder names the language(s):
``de/gur350.txt`` is monolingual German; ``en-de/rg65.txt`` pairs an English
word with a German one.

    python tools/get_similarity_data.py            # -> data/similarity/

The datasets keep their original licences; see the README for citations.
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

VECTO = "https://raw.githubusercontent.com/vecto-ai/word-benchmarks/9b61d0e067e2e68af22160d09450f82d9544cd70/word-similarity"
BENATHI = ("https://raw.githubusercontent.com/benathi/multisense-prob-fasttext/"
           "3dc7cb37af4bba68d0c2af453d4fb52966d1f061/data/word_similarity_datasets")
MULTI = BENATHI + "/wordsim_foreign_clean"
KAWAKAMI = "https://raw.githubusercontent.com/k-kawakami/embedding-evaluation/2a7f4cd93a1bdc11bd5b11368abbe51d786179c5/wordsim/data"


class Source(NamedTuple):
    dest: str  # path under data/similarity/
    url: str
    sep: Optional[str]  # None = whitespace
    cols: tuple  # (word1, word2, score) column indices
    citation: str


SOURCES: List[Source] = [
    # German: the datasets the original WE-Word-similarity-task used
    Source("de/gur65.txt", f"{BENATHI}/foreign/de/gur65.txt", ":", (0, 1, 2), "Gurevych 2005"),
    Source("de/gur350.txt", f"{BENATHI}/foreign/de/gur350.txt", ":", (0, 1, 2), "Gurevych 2005"),
    Source("de/zg222.txt", f"{BENATHI}/foreign/de/zg222.txt", ":", (0, 1, 2), "Zesch & Gurevych 2006"),
    Source("de/ws353.txt", f"{MULTI}/monolingual/WS353-ALL/MWS353_German.txt", ",", (0, 1, -1), "Leviant & Reichart 2015"),
    Source("de/simlex999.txt", f"{MULTI}/monolingual/SimLex/MSimLex999_German.txt", ",", (0, 1, -1), "Leviant & Reichart 2015"),
    # Spanish
    Source("es/ws353.txt", f"{KAWAKAMI}/es/ES-WS-353.txt", "\t", (0, 1, 2), "Hassan & Mihalcea 2009 (no accents)"),
    Source("es/rg65.txt", f"{MULTI}/monolingual/RG/rg65_spanish.txt", "\t", (0, 1, 2), "Camacho-Collados et al. 2015"),
    # Russian
    Source("ru/hj.txt", f"{VECTO}/monolingual/ru/hj.csv", ",", (1, 2, 3), "Panchenko et al. 2016"),
    Source("ru/ws353.txt", f"{MULTI}/monolingual/WS353-ALL/MWS353_Russian.txt", ",", (0, 1, -1), "Leviant & Reichart 2015"),
    Source("ru/simlex999.txt", f"{MULTI}/monolingual/SimLex/MSimLex999_Russian.txt", ",", (0, 1, -1), "Leviant & Reichart 2015"),
    # English
    Source("en/ws353.txt", f"{KAWAKAMI}/en/EN-WS-353-ALL.txt", "\t", (0, 1, 2), "Finkelstein et al. 2001"),
    Source("en/simlex999.txt", f"{VECTO}/monolingual/en/simlex999.csv", ",", (1, 2, 3), "Hill et al. 2015"),
    Source("en/rg65.txt", f"{KAWAKAMI}/en/EN-RG-65.txt", "\t", (0, 1, 2), "Rubenstein & Goodenough 1965"),
    Source("en/men.txt", f"{KAWAKAMI}/en/EN-MEN-TR-3k.txt", None, (0, 1, 2), "Bruni et al. 2014"),
    # Cross-lingual: word 1 in the first language, word 2 in the second
    Source("en-de/rg65.txt", f"{MULTI}/crosslingual/rg-crosslingual/rg65_EN-DE.txt", "\t", (0, 1, 2), "Camacho-Collados et al. 2015"),
    Source("en-de/ws353.txt", f"{MULTI}/crosslingual/ws353-crosslingual/cross_en_de_MWS353.txt", "\t", (0, 1, 2), "Leviant & Reichart 2015"),
    Source("en-de/simlex999.txt", f"{MULTI}/crosslingual/simlex-crosslingual/cross_en_de_SimLex.txt", "\t", (0, 1, 2), "Leviant & Reichart 2015"),
    Source("en-de/semeval17.txt", f"{VECTO}/cross-lingual/en-de/semeval17.csv", ",", (1, 2, 3), "Camacho-Collados et al. 2017"),
    Source("en-es/rg65.txt", f"{MULTI}/crosslingual/rg-crosslingual/rg65_EN-ES.txt", "\t", (0, 1, 2), "Camacho-Collados et al. 2015"),
    Source("en-es/semeval17.txt", f"{VECTO}/cross-lingual/en-es/semeval17.csv", ",", (1, 2, 3), "Camacho-Collados et al. 2017"),
]


def convert(text: str, src: Source) -> List[str]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(src.sep)]
        try:
            w1, w2, score = (parts[i] for i in src.cols)
            float(score)
        except (IndexError, ValueError):
            continue  # header line
        rows.append(f"{w1}\t{w2}\t{score}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/similarity")
    args = p.parse_args()
    out = Path(args.out)
    for src in SOURCES:
        with urllib.request.urlopen(src.url, timeout=60) as resp:
            text = resp.read().decode("utf-8-sig")
        rows = convert(text, src)
        dest = out / src.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{src.dest:<22} {len(rows):>5} pairs  ({src.citation})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
