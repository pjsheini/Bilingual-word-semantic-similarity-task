# Bilingual Word Semantic Similarity Task

Measures how well a **cross-lingual word-embedding model** puts a word and
its translation close together. For every word in a bilingual test
dictionary, the code looks up the nearest words in the other language and
checks whether a correct translation is among the top 1, 5 or 10
(precision@k, the standard MUSE word-translation benchmark).

The project was first written to test the *Bilingual Universal Concepts*
(BUC) model. Version 2 is a rewrite: plain Python 3.9+, NumPy only, and
fast. It works with any aligned embeddings in fastText `.vec` format.

## Quick start

```bash
pip install -r requirements.txt            # just numpy

# Evaluate one direction: separate aligned files per language...
python -m bwss evaluate --src en.vec --tgt es.vec --dict dicts/en-es.5000-6500.txt

# ...or one shared bilingual file, like the original BUC model
python -m bwss evaluate --shared buc_en-es.vec --dict dicts/en-es.5000-6500.txt

# Evaluate every dictionary in dicts/ against vectors/<lang>.vec|.npz
python -m bwss run-all --vectors-dir vectors

# Explore: nearest words and analogies, across languages
python -m bwss neighbours --vec vectors/en.npz --vec vectors/es.npz king es:rey
python -m bwss analogy    --vec vectors/en.npz --vec vectors/es.npz man king woman  hombre rey woman
```

`run-all` prints a table and writes `results/summary.md`,
`results/summary.json`, and one `predictions-<pair>-<method>.tsv` per run,
listing each word's gold translations, the rank of the first hit and the
top-10 candidates with scores.

### Options worth knowing

| Option | Default | Meaning |
|---|---|---|
| `--method nn\|csls\|both` | `both` | Cosine nearest neighbour, CSLS, or both |
| `--max-vocab N` | 200000 | Search the N most frequent words (MUSE standard); `0` = all |
| `--clean` | off | The original project's filter: drop 1-char/punctuation tokens and near-duplicates (`U.S.` / `US`) |
| `--lowercase` | off | Lowercase the vocabulary (the dictionaries are lowercase) |
| `--cache-dir` | `.cache` | Where CSLS statistics are cached between runs |

**CSLS** (Cross-domain Similarity Local Scaling, Conneau et al. 2018) reduces
the "hubness" problem, where a few words are close to everything. It
usually adds several points of accuracy. Its per-word statistics take a
few minutes to compute for a 200k vocabulary, so they are cached.

## Demo results

The BUC model files aren't public, so I built baseline embeddings from
public data (`tools/build_demo_vectors.py`) to show the pipeline running:

* 300-d vectors from spaCy's `*_lg` models, ranked by frequency with
  `wordfreq` (en 188k words, es 198k, de 112k, ru 200k).
* Spanish, German and Russian mapped onto English with an orthogonal
  (Procrustes) transform, learned **without a bilingual dictionary**.
  The seed is identically spelled words and numerals, refined by 8 rounds
  of self-learning. No word from any test dictionary is used in the seed.
* Evaluated on Python 3.9 with `--max-vocab 200000` (the default).

| Pair | Evaluated words | NN P@1 | NN P@5 | NN P@10 | CSLS P@1 | CSLS P@5 | CSLS P@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| en → es | 1,413 (94%) | 45.2 | 65.0 | 70.6 | **53.6** | 72.0 | 76.6 |
| es → en | 1,450 (97%) | 50.2 | 67.4 | 73.4 | **51.9** | 69.0 | 74.1 |
| en → de | 1,304 (87%) | 38.5 | 55.4 | 62.2 | **48.2** | 65.6 | 71.1 |
| de → en | 1,275 (85%) | 37.6 | 56.2 | 61.3 | **38.4** | 54.7 | 60.2 |
| en → ru | 1,469 (98%) | 28.0 | 44.2 | 49.6 | **34.3** | 53.0 | 59.6 |
| ru → en | 1,432 (95%) | **43.9** | 63.7 | 68.9 | 43.1 | 62.0 | 67.5 |

Precision in %. Full per-word output is in [`results/`](results/).

* CSLS helps most when translating **out of** English: +8 to +10 points
  P@1 for en→es and en→de. Into English it changes little.
* English–Russian is hardest. The two languages share almost no spellings,
  so the unsupervised seed is just 275 numerals and Latin-script tokens.
  It still ends up at 34–44% P@1.
* German coverage is lower (85–87%) because spaCy's German vectors cover
  only 112k words.

Precision is computed over test words whose source word and at least one
correct translation are in the vocabulary ("evaluated"). This is the MUSE
protocol.

For comparison, MUSE reports roughly 80% P@1 for en-es with fastText
Wikipedia vectors and supervised alignment. These baseline numbers are lower
because spaCy's vectors are noisier and the alignment is unsupervised. They're
a reference point for comparing other models, not a ceiling.

<details>
<summary>Example translations (en-es, CSLS)</summary>

| English | Gold | Rank of first hit | Top 5 candidates |
|---|---|---:|---|
| dancer | bailarina, bailarín | 1 | bailarín, bailarina, bailarin, coreógrafo, coreógrafa |
| phenomenon | fenómeno | 1 | fenómeno, epifenómeno, vulcanismo, meteorismo, fenónemo |
| novelists | novelistas | 1 | novelistas, escritores, historietistas, literatos, escritoras |
| abilities | habilidades, capacidades | 2 | habilidad, habilidades, destrezas, destreza, aptitudes |
| discover | descubrir, descubra, ... | 2 | descubrirás, descubra, redescubra, oportunidad, atractor |
| relocated | trasladó, trasladado | 8 | reubicado, relocalizado, desplazado, mudado, emplazado |
| arguing | discutiendo | – | argumente, argumentista, argumentario, argumento, argumentar |

Misses are often near-misses. Several are other inflections of the right word
(`descubrirás`), and some are valid synonyms the dictionary doesn't list
(`reubicado` for *relocated*).

Cross-lingual exploration with the same vectors:

```
$ python -m bwss analogy --vec vectors/en.npz --vec vectors/es.npz --vec vectors/de.npz \
    man king woman  hombre rey mujer  man king mujer  mann könig frau
man : king :: woman : queen      [queen [en] 0.633, prince [en] 0.584, lady [en] 0.582]
hombre : rey :: mujer : reina    [reina [es] 0.727, princesa [es] 0.645, monarca [es] 0.612]
man : king :: mujer : queen      [queen [en] 0.619, prince [en] 0.541, esposa [es] 0.527]
mann : könig :: frau : königin   [königin [de] 0.794, könige [de] 0.698, königs [de] 0.696]

$ python -m bwss neighbours --vec vectors/en.npz --vec vectors/es.npz --vec vectors/de.npz -n 6 es:rey
es:rey: monarca [es] 0.836, príncipe [es] 0.795, emperador [es] 0.725, vasallo [es] 0.715, trono [es] 0.710, könig [de] 0.706
```

</details>

## How it works

```
bwss/
  embeddings.py  load .vec/.npz, normalise, optional cleaning, save
  retrieval.py   batched top-k search with NumPy; CSLS with on-disk cache
  evaluate.py    dictionary loading, precision@k, per-word predictions
  explore.py     nearest neighbours and analogies across aligned spaces
  cli.py         python -m bwss ...
tools/build_demo_vectors.py   builds the demo embeddings above
tests/                        unit tests on synthetic data
dicts/                        MUSE test dictionaries
results/                      demo results (summary + per-word TSVs)
```

Each lookup is one matrix multiplication over L2-normalised vectors, done in
batches, so evaluating 1,500 words against a 200k-word vocabulary takes about
2 seconds. The original pure-Python loop took several seconds **per word**.

In a shared space (`--shared`, or `--src` and `--tgt` pointing at the same
file), a word can't return itself as its translation, but a gold translation
spelled the same as the source (`okay` → `okay`) counts as a rank-1 hit. This
is the same rule the original `translate.py` used.

### Test dictionaries

The `dicts/` files are the standard **MUSE** test sets from Facebook Research
([facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)). They
contain source words ranked 5000–6500 by frequency, one `source target` pair
per line. A word can have several correct translations. MUSE data is licensed
CC BY-NC 4.0.

| File | Pairs | Source words |
|---|---:|---:|
| `en-de.5000-6500.txt` | 3,660 | 1,500 |
| `de-en.5000-6500.txt` | 2,827 | 1,500 |
| `en-es.5000-6500.txt` | 2,975 | 1,500 |
| `es-en.5000-6500.txt` | 2,416 | 1,500 |
| `en-ru.5000-6500.txt` | 2,447 | 1,500 |
| `ru-en.5000-6500.txt` | 2,069 | 1,500 |

## Reproducing the demo

```bash
pip install numpy spacy wordfreq
mkdir models && cd models
for m in en_core_web_lg es_core_news_lg de_core_news_lg ru_core_news_lg; do
  curl -LO https://github.com/explosion/spacy-models/releases/download/$m-3.8.0/$m-3.8.0-py3-none-any.whl
  python -m zipfile -e $m-3.8.0-py3-none-any.whl .
done
cd ..
python tools/build_demo_vectors.py --models-dir models --out vectors   # ~5 min
python -m bwss run-all --vectors-dir vectors                          # ~25 min on 2 cores (CSLS stats, cached after)
python -m unittest discover -s tests -t .
```

## What changed from version 1

Version 1 (see git history) was a set of research scripts that couldn't run:

* `translate.py` was Python 2 only; the other scripts were Python 3.
* The neighbour script wrote a file format the scorer couldn't read, and it
  crashed at the end with an undefined variable.
* Paths were hard-coded Windows paths, and the language pair was fixed in
  the code.
* Search was a pure-Python loop over every word in the vocabulary.
* Accuracy was divided by the number of neighbour lists, not the number of
  test words.

All of that is replaced by the `bwss` package. The original cleaning rules
(drop punctuation-edged tokens, merge near-duplicates) are kept behind
`--clean`. The analogy filter that depended on the private BUC concept file
(`PreUni3Concepts-final.txt`) was dropped.

## Credits

Test dictionaries: Facebook Research MUSE. Version 1's loader and analogy
code were adapted from Martin Konicek's public word-vectors tutorial. Demo
vectors: spaCy models (MIT) and `wordfreq` (Apache 2.0 code, CC BY-SA data).
