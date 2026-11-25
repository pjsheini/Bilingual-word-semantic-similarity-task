# Cross-lingual Word Embeddings with Universal Concepts

Evaluates **cross-lingual word embeddings** with two standard tests:

1. **Word translation.** For every word in a bilingual test dictionary, look
   up the nearest words in the other language and check whether a correct
   translation is among the top 1, 5 or 10 (precision@k, the MUSE benchmark).
2. **Word similarity.** For word pairs that people have scored for similarity
   (German GUR350, Spanish WS353, English-German RG65, ...), check how well
   the vectors' cosine similarities rank the pairs the same way people did
   (Spearman's ρ). This covers both single languages and pairs of languages.
   It folds in the old
   [WE-Word-similarity-task](https://github.com/pjsheini/WE-Word-similarity-task)
   repository.

It can also **train** the model the project was built for: Bilingual
Universal Concepts (BUC) embeddings (Sheinidashtegol & Musaev, 2019). See
[Training BUC embeddings](#training-buc-embeddings).

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

# Word similarity: download the benchmarks once, then score every dataset
python tools/get_similarity_data.py                 # -> data/similarity/<lang>/*.txt
python -m bwss similarity --vectors-dir vectors     # or --shared buc_en-es.vec
```

`similarity` prints a table and writes `results/similarity.md` and
`results/similarity.json`. `run-all` prints a table and writes `results/summary.md`,
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

### Word similarity

Spearman's ρ between cosine similarity and human scores (1.0 means the same
ranking as people). Pairs with an out-of-vocabulary word are skipped, and
the coverage column shows how many were kept.

| Languages | Dataset | Pairs found | Coverage | Spearman ρ |
|---|---|---:|---:|---:|
| de | gur350 | 236/350 | 67.4% | 0.680 |
| de | gur65 | 44/65 | 67.7% | 0.708 |
| de | simlex999 | 857/999 | 85.8% | 0.332 |
| de | ws353 | 292/350 | 83.4% | 0.440 |
| de | zg222 | 122/222 | 55.0% | 0.438 |
| en | men | 3000/3000 | 100.0% | 0.782 |
| en | rg65 | 65/65 | 100.0% | 0.784 |
| en | simlex999 | 999/999 | 100.0% | 0.316 |
| en | ws353 | 353/353 | 100.0% | 0.668 |
| es | rg65 | 65/65 | 100.0% | 0.858 |
| es | ws353 | 345/352 | 98.0% | 0.548 |
| ru | hj | 389/398 | 97.7% | 0.666 |
| ru | simlex999 | 959/999 | 96.0% | 0.300 |
| ru | ws353 | 339/350 | 96.9% | 0.591 |
| en-de | rg65 | 103/125 | 82.4% | 0.705 |
| en-de | semeval17 | 560/914 | 61.3% | 0.613 |
| en-de | simlex999 | 1477/1595 | 92.6% | 0.278 |
| en-de | ws353 | 579/638 | 90.8% | 0.523 |
| en-es | rg65 | 126/126 | 100.0% | 0.825 |
| en-es | semeval17 | 730/978 | 74.6% | 0.684 |

* **Within each language,** the vectors do well on relatedness sets (MEN
  0.78, RG65 0.78–0.86, GUR65 0.71) and worse on SimLex-999 (about 0.3).
  SimLex scores strict similarity, and plain distributional vectors are
  known to blur that with relatedness (*cup* ~ *coffee*).
* **Across languages,** the aligned spaces keep most of the monolingual
  quality. en–es RG65 is 0.83, close to the Spanish-only 0.86, so the
  alignment doesn't destroy within-language structure.
* **German coverage is low** (55–68% on GUR/ZG). The German demo vocabulary
  is only 112k words and these datasets are full of rare compounds. This is
  a limit of the demo vectors, not the method.
* **Spanish WS353 is distributed without accents** (`quimica`).
  `--fold-accents` matches words ignoring accents, but the demo vectors
  already contain the unaccented web spellings, so it's off by default.
  With it on, ρ is 0.537 instead of 0.548.

## Training BUC embeddings

`python -m bwss buc` implements the method of the paper, *Learning
Cross-lingual Word Embeddings with Universal Concepts* (Sheinidashtegol &
Musaev, IJWSC 10(3), 2019, [doi:10.5121/ijwsc.2019.10302](https://doi.org/10.5121/ijwsc.2019.10302)).
It replaces the scripts in
[MUC-Encoding-decoding-](https://github.com/pjsheini/MUC-Encoding-decoding-) and
[UCGenerator-Clusterring](https://github.com/pjsheini/UCGenerator-Clusterring).

| Step | What happens | Command | Old script |
|---|---|---|---|
| 1. Concepts | Bilingual dictionaries become a graph (words are nodes, translations are edges). Each connected component is one universal concept. | `buc concepts` | `UConceptByClustering.py` |
| 2. Rewrite | Each concept word in each monolingual corpus is replaced by its concept ID. | `buc rewrite` | `words2uuid.py` |
| 3. Train | word2vec or fastText on all rewritten corpora together (window 10, as in the paper). | `buc train` | (not in the repos) |
| 4. Expand | Every word in a concept gets the concept's vector. One aligned `<lang>.npz` per language, plus `shared.vec` with `--shared`. | `buc train` | `uuid2word.py` |

`buc pipeline` runs all four steps. On your own machine:

```bash
pip install -r requirements-buc.txt      # gensim

# MUSE *training* dictionaries (0-5000). The test sets in dicts/ are 5000-6500.
for p in en-es es-en; do
  curl -O https://dl.fbaipublicfiles.com/arrival/dictionaries/$p.0-5000.txt
done

# Corpora: plain text, one sentence or paragraph per line, tokenised
# (e.g. Wikipedia dumps through wikiextractor). The paper used en + es Wikipedia.
python -m bwss buc pipeline \
    --dict en-es.0-5000.txt --dict es-en.0-5000.txt \
    --exclude dicts/en-es.5000-6500.txt --exclude dicts/es-en.5000-6500.txt \
    --corpus en:enwiki.txt --corpus es:eswiki.txt \
    --out vectors_buc --shared

python -m bwss run-all    --vectors-dir vectors_buc   # word translation
python -m bwss similarity --vectors-dir vectors_buc   # word similarity
```

Add more languages with more dictionaries and `--corpus` entries. Concepts
then span all of them; this is the paper's multilingual MUC variant.
`buc_stats.json` records the concept counts, the share of corpus tokens
replaced, and the training time.

**Things that differ from the original scripts, on purpose:**

* **Test words are kept out of concepts** (`--exclude`). If a test pair were
  inside a concept, the two words would share one vector by construction,
  and the translation test would measure the dictionary, not the
  embeddings. The paper reports about 35k source words per pair, which
  sounds like the full MUSE dictionaries (these include the test range).
  Use the 0–5000 files, or the full ones plus `--exclude`.
* **Words are tracked per language.** English *no* and Spanish *no* only
  join if a dictionary pair links them. Other tokens are written as
  `en:word` / `es:word` in the rewritten corpus, so the two vocabularies
  don't collide in the shared model.
* **The corpus is rewritten line by line.** `words2uuid.py` read 1 KB
  chunks, which split words at chunk edges. `uuid2word.py` wrote each
  vector as `b'0.12 …'` on Python 3, which made the output unloadable.
  Both are fixed.
* **Optional `--max-concept-size N`.** Connected components can chain
  loosely related words into very large concepts. The paper keeps them
  all, and so does the default.
* **Concept files from the original project** (`UC_Nodelist`/`GUID` JSON
  lines) can be reused with `--legacy-langs en es`.
* **The Google Translate + OPTICS clustering variant in UCGenerator-Clusterring
  is not ported.** It isn't the method in the paper, and it depended on
  scraping the Google Translate website.

Words in one concept have *identical* vectors, so when a concept has several
words in the target language, their order in a top-k list is a tie. This is
inherent to the method and is worth keeping in mind when reading P@1.

**Reported results, for comparison.** The paper's bilingual test counts a
pair as correct when the translation is among the 10 nearest neighbours,
which is P@10 here. BUCWE scored 40.95 (en→es), 53.68 (es→en),
52.53 (en→de), 33.53 (de→en). On word similarity it reported 0.67 on
GUR350 and 0.60 on Spanish WS353. The demo baseline above gets
76.6 / 74.1 / 71.1 / 60.2 P@10 (CSLS) and 0.68 / 0.55. The setups differ,
so only a rerun with this pipeline gives a like-for-like comparison.

The pipeline is tested end to end on a synthetic two-language corpus
(`tests/test_buc.py`). Words held out of every concept still land among
their translation's topic words (P@5 = 100%), which shows the shared
concept contexts carry meaning across languages. This workspace couldn't
download Wikipedia or the MUSE training dictionaries, so a full-scale run
is still to do.

## How it works

```
bwss/
  embeddings.py  load .vec/.npz, normalise, optional cleaning, save
  retrieval.py   batched top-k search with NumPy; CSLS with on-disk cache
  evaluate.py    dictionary loading, precision@k, per-word predictions
  explore.py     nearest neighbours and analogies across aligned spaces
  similarity.py  word-similarity datasets, Spearman correlation
  buc.py         BUC training: concepts, corpus rewrite, word2vec/fastText, expand
  cli.py         python -m bwss ...
tools/build_demo_vectors.py   builds the demo embeddings above
tools/get_similarity_data.py  downloads the similarity benchmarks (pinned commits)
tests/                        unit tests on synthetic data
dicts/                        MUSE test dictionaries
results/                      demo results (translation summary, per-word TSVs,
                              similarity.md/json)
```

Each lookup is one matrix multiplication over L2-normalised vectors, done in
batches, so evaluating 1,500 words against a 200k-word vocabulary takes about
2 seconds. The original pure-Python loop took several seconds **per word**.

In a shared space (`--shared`, or `--src` and `--tgt` pointing at the same
file), a word can't return itself as its translation, but a gold translation
spelled the same as the source (`okay` → `okay`) counts as a rank-1 hit. This
is the same rule the original `translate.py` used.

### Word-similarity datasets

`tools/get_similarity_data.py` downloads 20 datasets from pinned commits of
public GitHub mirrors and rewrites each as `word1<TAB>word2<TAB>score`. The
parent folder names the language(s), so `en-de/rg65.txt` pairs an English
word with a German one. `bwss similarity` also reads the original files
directly (tab, `:`-separated GUR format, comma or space separated). The
datasets keep their own licences, so they aren't stored in this repo.

| Languages | Datasets | Source |
|---|---|---|
| de | GUR65, GUR350 (Gurevych 2005), ZG222 (Zesch & Gurevych 2006), WS353, SimLex-999 (Leviant & Reichart 2015) | UKP TU Darmstadt; Multilingual WS353/SimLex |
| es | WS353 (Hassan & Mihalcea 2009), RG65 (Camacho-Collados et al. 2015) | |
| ru | HJ (Panchenko et al. 2016), WS353, SimLex-999 (Leviant & Reichart 2015) | |
| en | WS353, SimLex-999, RG65, MEN | |
| en-de | RG65, WS353, SimLex-999, SemEval-2017 | cross-lingual |
| en-es | RG65, SemEval-2017 | cross-lingual |

Mirrors: [vecto-ai/word-benchmarks](https://github.com/vecto-ai/word-benchmarks),
[benathi/multisense-prob-fasttext](https://github.com/benathi/multisense-prob-fasttext),
[k-kawakami/embedding-evaluation](https://github.com/k-kawakami/embedding-evaluation).

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
python tools/get_similarity_data.py && python -m bwss similarity      # ~1 min
python -m unittest discover -s tests -t .   # BUC end-to-end test runs if gensim is installed
```

## What changed from version 1

Version 1 (see git history) was a set of research scripts that couldn't run:

* All scripts wrote in Python 2 only; 
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

The word-similarity test replaces the separate `WE-Word-similarity-task`
repository. That script only wrote `human score : cosine` lines and never
computed the correlation. It imported modules that lived in this repo,
and it handled one hard-coded dataset at a time.

## Credits

Test dictionaries: Facebook Research MUSE. Word-similarity datasets: the
authors cited above, via the mirrors listed. Version 1's loader and analogy
code were adapted from Martin Konicek's public word-vectors tutorial. Demo
vectors: spaCy models (MIT) and `wordfreq` (Apache 2.0 code, CC BY-SA data).
