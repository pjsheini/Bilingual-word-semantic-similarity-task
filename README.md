# Bilingual Word Semantic Similarity Task

Scripts for testing how well a **bilingual word-embedding model** places a word
and its translation close together in a shared vector space.

The model under test is the *Bilingual Universal Concepts* (BUC) model: a
cross-lingual embedding where words from two languages (e.g. English and
Spanish) live in one 300-dimensional space. If the space is well aligned, the
nearest neighbours of an English word should include its Spanish translation.

> **Status:** archived research code. The embedding files and concept files
> the scripts expect are **not** in this repository, so the scripts cannot be
> run as they are. See [Known issues](#known-issues).

## How the evaluation works

1. **Load the embeddings.** `load.py` reads a fastText-style `.vec` file
   (one word per line, followed by 300 floats), L2-normalises every vector,
   drops tokens that are too short or start/end with punctuation, and removes
   near-duplicates (`U.S.` / `US`).
2. **Find nearest neighbours.** For every source word in a bilingual test
   dictionary, `WE-10MostSimilarWords-MT.py` ranks all vocabulary words by
   cosine similarity and writes the 10 closest to a results file.
3. **Score translation accuracy.** `translate.py` reads that neighbour list
   and checks whether any gold translation from the dictionary appears in the
   top-k neighbours. It reports hits per rank and an overall
   precision@k-style percentage.

`main.py` is an exploration script: it prints nearest neighbours and solves
analogies (`man : king :: woman : ?`) within and across languages
(`hombre : rey :: woman : ?`), optionally mapping words to shared concept
IDs from a BUC concept file.

## Repository contents

| File | Purpose |
|---|---|
| `load.py` | Load and clean a `.vec` embedding file |
| `vectors.py` | Pure-Python vector maths (dot, add, sub, normalise, cosine) |
| `word.py` | `Word` class: text, vector, frequency rank |
| `main.py` | Nearest-neighbour and analogy exploration, incl. cross-lingual analogies |
| `WE-10MostSimilarWords-MT.py` | Writes the top-10 nearest neighbours for every word in a test dictionary |
| `translate.py` | Scores those neighbour lists against the gold translations |
| `*.5000-6500.txt` | Bilingual test dictionaries (see below) |

### Test dictionaries

The dictionaries are the standard **MUSE** bilingual test sets from Facebook
Research ([facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)),
covering source words ranked 5000–6500 by frequency. Each line is
`source_word target_word`, and a source word can have more than one valid
translation.

| File | Pairs |
|---|---|
| `en-de.5000-6500.txt` | 3,660 |
| `de-en.5000-6500.txt` | 2,827 |
| `en-es.5000-6500.txt` | 2,975 |
| `es-en.5000-6500.txt` | 2,416 |
| `en-ru.5000-6500.txt` | 2,447 |
| `ru-en.5000-6500.txt` | 2,069 |

MUSE data is distributed under its own licence (CC BY-NC 4.0); check it
before reusing these files.

## Data the scripts expect (not included)

| Path in code | What it is |
|---|---|
| `data\final_en-es_clustering_50k.vec` | Aligned EN–ES BUC embeddings, 50k vocabulary |
| `data\finalWordENESPT.vec` | Aligned EN–ES–PT embeddings (used by `main.py`) |
| `PreUni3Concepts-final.txt` | JSON-lines file mapping concept GUIDs to word lists |
| `UnSup-vectors-en.txt` | Neighbour list that `translate.py` scores |
| `dicts\en-es.5000-6500.txt` | Same as the root-level `en-es` file, expected in a `dicts\` folder |

Any 300-dimensional, cross-lingually aligned `.vec` file (for example the
aligned fastText vectors published by MUSE) should work in place of the BUC
files.

## Running (once the data is in place)

```bash
# 1. Nearest neighbours for every source word (slow: pure-Python brute force)
python WE-10MostSimilarWords-MT.py

# 2. Score them (translate.py is Python 2 code; see below)
python2 translate.py

# Exploration: neighbours and analogies
python main.py
```

## Known issues

- **Paths are Windows-style and hard-coded** (`data\\...`, `dicts\\...`), and
  the language pair is fixed in the code. Change the paths to switch pairs.
- **`translate.py` only runs on Python 2.** It calls `.decode('unicode-escape')`
  on `str`, which fails on Python 3. The other scripts are Python 3
  (f-strings, type hints).
- **`WE-10MostSimilarWords-MT.py` crashes at the end:** it prints a timing
  using `t2`, which is only defined inside a commented-out threading block.
  The results file has already been written by then.
- **The neighbour file and the scorer do not line up.** Step 2 writes
  `similarword-ENES-clusterring50k.txt` as `word<TAB>[(score, word), ...]`,
  but `translate.py` reads `UnSup-vectors-en.txt` and splits on spaces. Some
  conversion step was used between them and is not in the repo.
- **Accuracy is divided by the number of neighbour lists, not by the number
  of test words,** so words missing from the vocabulary are not counted as
  misses.
- **The search is slow.** Every query is a pure-Python scan over the whole
  vocabulary. NumPy matrix multiplication would make it orders of magnitude
  faster.
- In `main.py`, the analogy redundancy filter reloads both concept files for
  every candidate word.

## Credits

`load.py`, `vectors.py`, `word.py` and the analogy logic in `main.py` look
adapted from Martin Konicek's public *word vectors* tutorial code. The test
dictionaries come from Facebook Research's MUSE project.
