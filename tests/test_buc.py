"""Tests for the BUC pipeline: ``python -m unittest discover -s tests -t .``"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bwss import buc
from bwss.cli import main
from bwss.embeddings import Embeddings
from bwss.evaluate import evaluate, load_dictionary

try:
    import gensim  # noqa: F401
    HAVE_GENSIM = True
except ImportError:
    HAVE_GENSIM = False


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestConcepts(unittest.TestCase):
    def test_components_merge_across_dictionaries(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            en_es = _write(d / "en-es.txt", "fox zorro\ndog perro\ndog can\n")
            de_en = _write(d / "de-en.txt", "Fuchs fox\nHund dog\n")
            concepts, stats = buc.build_concepts([en_es, de_en])
            self.assertEqual(stats["concepts"], 2)
            fox = concepts.concept_of("en", "fox")
            self.assertEqual(concepts.members[fox], {"de": ["fuchs"], "en": ["fox"], "es": ["zorro"]})
            self.assertEqual(sorted(concepts.members[concepts.concept_of("es", "perro")]["es"]), ["can", "perro"])
            self.assertIsNone(concepts.concept_of("es", "fox"))  # languages stay separate

    def test_exclude_keeps_test_words_out(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            train = _write(d / "en-es.0-5000.txt", "fox zorro\ncat gato\ncat minino\n")
            test = _write(d / "en-es.5000-6500.txt", "cat gato\n")
            concepts, stats = buc.build_concepts([train], exclude=[test])
            self.assertIsNone(concepts.concept_of("en", "cat"))
            self.assertIsNone(concepts.concept_of("es", "gato"))
            self.assertIsNone(concepts.concept_of("es", "minino"))  # its only link was "cat"
            self.assertEqual(stats["pairs_excluded"], 2)

    def test_max_size_and_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            dct = _write(d / "en-es.txt", "a1 b1\na1 b2\na1 b3\nx y\n")
            concepts, stats = buc.build_concepts([dct], max_size=2)
            self.assertEqual((stats["concepts"], stats["dropped_large_concepts"]), (1, 1))
            concepts.save(d / "c.jsonl")
            again = buc.Concepts.load(d / "c.jsonl")
            self.assertEqual(again.members, concepts.members)

    def test_load_original_project_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d) / "old.txt", json.dumps(
                {"UC_Nodelist": [["fox", "zorro"]], "UC_edgelist": [[]], "GUID": ["abc"], "color": ["#fff"]}) + "\n")
            c = buc.Concepts.load(p, legacy_langs=["en", "es"])
            self.assertEqual(c.concept_of("es", "zorro"), "UC_abc")
            with self.assertRaises(ValueError):
                buc.Concepts.load(p)

    def test_rewrite_and_expand(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            concepts, _ = buc.build_concepts([_write(d / "en-es.txt", "fox zorro\n")])
            src = _write(d / "en.txt", "The fox runs\n\n")
            stats = buc.rewrite_corpus(concepts, "en", src, d / "out.txt")
            cid = concepts.concept_of("en", "fox")
            self.assertEqual((d / "out.txt").read_text(encoding="utf-8"), f"en:the {cid} en:runs\n")
            self.assertEqual((stats["tokens"], stats["replaced"]), (3, 1))
            tokens = [cid, "en:the", "es:el", "unknown"]
            vecs = np.eye(4, dtype=np.float32)
            spaces = buc.expand(tokens, vecs, concepts, ["en", "es"])
            self.assertEqual(spaces["en"].words, ["fox", "the"])
            self.assertEqual(spaces["es"].words, ["zorro", "el"])
            self.assertTrue(np.allclose(spaces["en"].vector("fox"), spaces["es"].vector("zorro")))
            self.assertEqual(len(buc.shared_space(spaces)), 4)


def _toy_world(d: Path, seed: int = 0):
    """Two languages, 12 topics x 5 meanings. Meanings 0-2 of each topic are
    in the training dictionary; 3-4 are test words, never in a concept."""
    rng = random.Random(seed)
    topics, per = 12, 5
    train_lines, test_lines = [], []
    for t in range(topics):
        for m in range(per):
            pair = f"en{t}x{m} es{t}x{m}"
            (train_lines if m < 3 else test_lines).append(pair)
    train = _write(d / "dicts" / "en-es.0-5000.txt", "\n".join(train_lines) + "\n")
    test = _write(d / "dicts" / "en-es.5000-6500.txt", "\n".join(test_lines) + "\n")
    for lang in ("en", "es"):
        lines = []
        for _ in range(3000):
            t = rng.randrange(topics)
            lines.append(" ".join(f"{lang}{t}x{rng.randrange(per)}" for _ in range(8)))
        _write(d / f"{lang}.txt", "\n".join(lines) + "\n")
    return train, test


@unittest.skipUnless(HAVE_GENSIM, "gensim not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def test_pipeline_transfers_to_held_out_words(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            train, test = _toy_world(d)
            code = main(["buc", "pipeline", "--dict", str(train), "--exclude", str(test),
                         "--corpus", f"en:{d / 'en.txt'}", "--corpus", f"es:{d / 'es.txt'}",
                         "--work-dir", str(d / "work"), "--out", str(d / "vectors"),
                         "--dim", "32", "--min-count", "1", "--epochs", "5", "--workers", "1", "--shared"])
            self.assertEqual(code, 0)
            en = Embeddings.load(d / "vectors" / "en.npz")
            es = Embeddings.load(d / "vectors" / "es.npz")
            self.assertTrue((d / "vectors" / "shared.vec").exists())
            stats = json.loads((d / "vectors" / "buc_stats.json").read_text())
            self.assertEqual(stats["concepts"], 36)
            # Held-out words share no vector with their translation, yet the
            # shared concept contexts put them in the right topic: with 5
            # Spanish words per topic, the translation should be in the top 5
            # far more often than chance (5/60).
            result = evaluate(en, es, load_dictionary(test), ks=(1, 5))
            self.assertEqual(result.evaluated_words, 24)
            self.assertGreater(result.precision[5], 60.0)


if __name__ == "__main__":
    unittest.main()
