import sys
from pathlib import Path
import numpy as np
import pytest
import requests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from embeddings import OllamaEmbedder, TfidfEmbedder, get_embedder, _l2_normalize
from config import CFG
TOY_DOCS = ['the stock rallied hard after earnings beat expectations', 'options traders piled into calls ahead of the print', 'management guidance was weak and shares dropped sharply', 'buy the dip this is a generational opportunity', 'revenue and margins both expanded year over year']

def test_tfidf_shape_and_norms():
    emb = TfidfEmbedder(dim=384)
    V = emb.encode(TOY_DOCS)
    assert V.shape == (5, 384)
    assert V.dtype == np.float32
    norms = np.linalg.norm(V, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-05)

def test_tfidf_deterministic():
    a = TfidfEmbedder(dim=384).encode(TOY_DOCS)
    b = TfidfEmbedder(dim=384).encode(TOY_DOCS)
    assert np.allclose(a, b, atol=1e-06)

def test_tfidf_empty_string_is_zero_vector():
    emb = TfidfEmbedder(dim=384)
    docs = TOY_DOCS[:2] + [''] + TOY_DOCS[2:]
    V = emb.encode(docs)
    assert V.shape == (6, 384)
    assert np.allclose(V[2], 0.0)
    other = np.linalg.norm(np.delete(V, 2, axis=0), axis=1)
    assert np.allclose(other, 1.0, atol=1e-05)

def test_tfidf_all_empty_inputs():
    emb = TfidfEmbedder(dim=384)
    V = emb.encode(['', '', ''])
    assert V.shape == (3, 384)
    assert np.allclose(V, 0.0)

def test_tfidf_empty_list():
    emb = TfidfEmbedder(dim=384)
    V = emb.encode([])
    assert V.shape == (0, 384)

def test_tfidf_smaller_dim():
    emb = TfidfEmbedder(dim=8)
    V = emb.encode(TOY_DOCS)
    assert V.shape == (5, 8)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-05)

def test_disk_cache_identical(tmp_path):
    emb = TfidfEmbedder(dim=64, use_cache=True, cache_root=tmp_path)
    emb.fit(TOY_DOCS)
    first = emb.encode(TOY_DOCS)
    second = emb.encode(TOY_DOCS)
    assert np.array_equal(first, second)
    files = list(Path(tmp_path).rglob('*.npy'))
    assert len(files) >= 1

def test_cache_survives_new_instance(tmp_path):
    emb1 = TfidfEmbedder(dim=64, use_cache=True, cache_root=tmp_path)
    emb1.fit(TOY_DOCS)
    first = emb1.encode(TOY_DOCS)
    emb2 = TfidfEmbedder(dim=64, use_cache=True, cache_root=tmp_path)
    emb2.fit(TOY_DOCS)
    second = emb2.encode(TOY_DOCS)
    assert np.array_equal(first, second)

def test_l2_normalize_zero_row_safe():
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(m)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)

def test_l2_normalize_1d():
    out = _l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert out.shape == (1, 2)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)

def test_get_embedder_tfidf_backend():
    import copy
    cfg = copy.copy(CFG)
    cfg.embed_backend = 'tfidf'
    emb = get_embedder(cfg)
    assert isinstance(emb, TfidfEmbedder)
    assert emb.dim == cfg.embed_dim

def test_get_embedder_ollama_degrades_when_down(monkeypatch):
    import copy
    monkeypatch.setattr(OllamaEmbedder, 'server_up', staticmethod(lambda *a, **k: False))
    cfg = copy.copy(CFG)
    cfg.embed_backend = 'ollama'
    emb = get_embedder(cfg)
    assert isinstance(emb, TfidfEmbedder)

def _ollama_up() -> bool:
    try:
        return requests.get('http://localhost:11434/api/tags', timeout=3).status_code == 200
    except Exception:
        return False

@pytest.mark.skipif(not _ollama_up(), reason='ollama server not reachable on localhost:11434')
def test_ollama_live_encode():
    emb = OllamaEmbedder(model='all-minilm', dim=384, use_cache=False)
    V = emb.encode(['hello world', 'earnings surprise drove the stock up', ''])
    assert V.shape == (3, 384)
    assert V.dtype == np.float32
    assert np.allclose(np.linalg.norm(V[0]), 1.0, atol=0.0001)
    assert np.allclose(np.linalg.norm(V[1]), 1.0, atol=0.0001)
    assert np.allclose(V[2], 0.0)
