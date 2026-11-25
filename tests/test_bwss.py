"""Unit tests on tiny synthetic data: ``python -m unittest discover tests``."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bwss import Embeddings, MultiSpace, evaluate, load_dictionary
from bwss.cli import main
from bwss.embeddings import canonical_form, is_valid_word
from bwss.retrieval import mean_knn_similarity, search


def _toy_spaces(seed: int = 0, n: int = 50, dim: int = 16):
    """English words and 'translations' that are the same vectors + small noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, dim)).astype(np.float32)
    en = [f"word{i}" for i in range(n)]
    xx = [f"palabra{i}" for i in range(n)]
    noisy = base + 0.05 * rng.normal(size=base.shape).astype(np.float32)
    return Embeddings(en, base, "en"), Embeddings(xx, noisy, "es")


class TestCleaning(unittest.TestCase):
    def test_stop_word_rule_matches_original(self):
        words = ["a", "ab", "-ab", "ab_", "a.", ".a", "ab"]
        self.assertEqual([w for w in words if is_valid_word(w)], ["ab", "ab"])

    def test_canonical_form(self):
        self.assertEqual(canonical_form("U.S."), canonical_form("US_"))

    def test_load_vec_with_header_clean_and_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.vec"
            p.write_text("6 2\nU.S 1 0\nUS 0 1\nU.S. 2 1\n-x 1 1\nnew york 3 4\nok 0 2\n", encoding="utf-8")
            emb = Embeddings.load(p, clean=True)
            # "US" duplicates "U.S"; "U.S." and "-x" fail the punctuation rule.
            self.assertEqual(emb.words, ["U.S", "new york", "ok"])
            self.assertTrue(np.allclose(np.linalg.norm(emb.vectors, axis=1), 1))
            self.assertEqual(len(Embeddings.load(p, max_vocab=2)), 2)

    def test_npz_round_trip(self):
        en, _ = _toy_spaces()
        with tempfile.TemporaryDirectory() as d:
            en.save(Path(d) / "en.npz")
            en.save(Path(d) / "en.vec")
            a = Embeddings.load(Path(d) / "en.npz")
            b = Embeddings.load(Path(d) / "en.vec")
            self.assertEqual(a.words, en.words)
            self.assertTrue(np.allclose(a.vectors, b.vectors, atol=1e-4))


class TestRetrieval(unittest.TestCase):
    def test_search_finds_identical_rows(self):
        en, _ = _toy_spaces()
        idx, scores = search(en.vectors, en.vectors, k=3)
        self.assertTrue((idx[:, 0] == np.arange(len(en))).all())
        self.assertTrue(np.all(np.diff(scores, axis=1) <= 1e-6))

    def test_batching_does_not_change_results(self):
        en, es = _toy_spaces()
        a = search(en.vectors, es.vectors, k=5, batch=7)[0]
        b = search(en.vectors, es.vectors, k=5, batch=1000)[0]
        self.assertTrue((a == b).all())

    def test_csls_requires_penalties(self):
        en, es = _toy_spaces()
        with self.assertRaises(ValueError):
            search(en.vectors, es.vectors, method="csls")

    def test_mean_knn_similarity_brute_force(self):
        en, es = _toy_spaces()
        sims = en.vectors @ es.vectors.T
        expected = np.sort(sims, axis=1)[:, -4:].mean(axis=1)
        self.assertTrue(np.allclose(mean_knn_similarity(en.vectors, es.vectors, 4, batch=9), expected))


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.en, self.es = _toy_spaces()
        self.dictionary = {f"word{i}": [f"palabra{i}"] for i in range(50)}
        self.dictionary["missing"] = ["palabra1"]

    def test_perfect_alignment_scores_100(self):
        for method in ("nn", "csls"):
            r = evaluate(self.en, self.es, self.dictionary, method=method)
            self.assertEqual(r.evaluated_words, 50)
            self.assertEqual(r.dictionary_words, 51)
            self.assertAlmostEqual(r.precision[1], 100.0)

    def test_wrong_gold_scores_0(self):
        wrong = {f"word{i}": [f"palabra{(i + 1) % 50}"] for i in range(50)}
        r = evaluate(self.en, self.es, wrong, ks=(1,))
        self.assertLess(r.precision[1], 10)

    def test_shared_space_excludes_self_and_counts_identity(self):
        en = self.en
        r = evaluate(en, en, {"word3": ["word3"], "word4": ["word5"]}, ks=(1, 10))
        by_word = {p.source: p for p in r.predictions}
        self.assertEqual(by_word["word3"].hit_rank, 1)
        self.assertNotIn("word4", by_word["word4"].predicted)

    def test_load_dictionary_groups_translations(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "en-es.txt"
            p.write_text("waves ola\nwaves olas\nwaves ola\nok ok\n\n", encoding="utf-8")
            self.assertEqual(load_dictionary(p), {"waves": ["ola", "olas"], "ok": ["ok"]})


class TestExploreAndCli(unittest.TestCase):
    def test_cross_lingual_neighbours_and_analogy(self):
        en, es = _toy_spaces()
        space = MultiSpace([en, es])
        self.assertEqual(space.neighbours("word7", 1)[0].word, "palabra7")
        self.assertIsNone(space.analogy("word1", "nope", "word2"))
        self.assertEqual(space.find("es:palabra2")[0].name, "es")

    def test_find_prefers_space_where_word_is_frequent(self):
        rng = np.random.default_rng(1)
        en = Embeddings(["the", "of", "mujer"], rng.normal(size=(3, 4)), "en")
        es = Embeddings(["mujer", "de"], rng.normal(size=(2, 4)), "es")
        space = MultiSpace([en, es])
        self.assertEqual(space.find("mujer")[0].name, "es")
        self.assertEqual(space.find("en:mujer")[0].name, "en")

    def test_cli_run_all(self):
        en, es = _toy_spaces()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            en.save(d / "vectors" / "en.npz")
            es.save(d / "vectors" / "es.vec")
            (d / "dicts").mkdir()
            (d / "dicts" / "en-es.5000-6500.txt").write_text(
                "\n".join(f"word{i} palabra{i}" for i in range(50)), encoding="utf-8")
            code = main(["run-all", "--vectors-dir", str(d / "vectors"), "--dicts-dir", str(d / "dicts"),
                         "--out-dir", str(d / "results"), "--cache-dir", str(d / "cache")])
            self.assertEqual(code, 0)
            self.assertTrue((d / "results" / "summary.md").exists())
            self.assertTrue((d / "results" / "predictions-en-es-csls.tsv").exists())


if __name__ == "__main__":
    unittest.main()
