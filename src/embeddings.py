from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, INTERIM

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)

def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

class _DiskCache:

    def __init__(self, namespace: str, dim: int, root: Optional[Path]=None):
        base = Path(root) if root is not None else INTERIM / 'emb_cache'
        safe = ''.join((c if c.isalnum() or c in '-_.' else '_' for c in namespace))
        self.dir = base / safe
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)

    def _path(self, text_key: str) -> Path:
        return self.dir / f'{text_key}.npy'

    def get(self, text: str) -> Optional[np.ndarray]:
        p = self._path(_text_key(text))
        if not p.exists():
            return None
        try:
            v = np.load(p)
        except Exception:
            return None
        if v.shape != (self.dim,):
            return None
        return v.astype(np.float32, copy=True)

    def put(self, text: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape != (self.dim,):
            return
        p = self._path(_text_key(text))
        tmp = p.with_suffix('.npy.tmp')
        try:
            np.save(tmp, vec)
            tmp.replace(p)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

def _cached_encode(texts: list[str], dim: int, cache: Optional[_DiskCache], compute) -> np.ndarray:
    n = len(texts)
    out = np.zeros((n, dim), dtype=np.float32)
    if n == 0:
        return out
    to_compute_idx: list[int] = []
    to_compute_txt: list[str] = []
    for i, t in enumerate(texts):
        if t is None or t == '':
            continue
        if cache is not None:
            hit = cache.get(t)
            if hit is not None:
                out[i] = hit
                continue
        to_compute_idx.append(i)
        to_compute_txt.append(t)
    if to_compute_txt:
        vecs = compute(to_compute_txt)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape != (len(to_compute_txt), dim):
            raise ValueError(f'backend returned shape {vecs.shape}, expected {(len(to_compute_txt), dim)}')
        for j, i in enumerate(to_compute_idx):
            out[i] = vecs[j]
            if cache is not None:
                cache.put(to_compute_txt[j], vecs[j])
    return out

class OllamaEmbedder:

    def __init__(self, model: str='all-minilm', dim: int=384, base_url: str='http://localhost:11434', batch: int=256, cfg=CFG, use_cache: bool=True, cache_root: Optional[Path]=None):
        self.model = model
        self.dim = int(dim)
        self.base_url = base_url.rstrip('/')
        self.batch = int(batch)
        self.cfg = cfg
        self.name = f'ollama:{model}'
        self._cache = _DiskCache(f'ollama_{model}_{self.dim}', self.dim, root=cache_root) if use_cache else None

    @staticmethod
    def server_up(base_url: str='http://localhost:11434', timeout: float=3.0) -> bool:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    @retry(retry=retry_if_exception_type((requests.RequestException, RuntimeError)), stop=stop_after_attempt(CFG.max_retries), wait=wait_exponential(multiplier=CFG.backoff_base, min=1, max=30), reraise=True)
    def _embed_call(self, batch_texts: list[str]) -> np.ndarray:
        resp = requests.post(f'{self.base_url}/api/embed', json={'model': self.model, 'input': batch_texts}, headers={'User-Agent': self.cfg.user_agent}, timeout=self.cfg.http_timeout)
        if resp.status_code == 429:
            raise RuntimeError('ollama 429 rate-limited')
        resp.raise_for_status()
        data = resp.json()
        embs = data.get('embeddings')
        if not embs or len(embs) != len(batch_texts):
            raise RuntimeError(f'ollama returned {(0 if not embs else len(embs))} embeddings for {len(batch_texts)} inputs')
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise RuntimeError(f'ollama embedding dim {arr.shape} != expected (_, {self.dim})')
        return arr

    def _compute(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start:start + self.batch]
            rows.append(self._embed_call(chunk))
        arr = np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(arr)

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        return _cached_encode(texts, self.dim, self._cache, self._compute)

class SentenceTransformerEmbedder:

    def __init__(self, model: str='all-MiniLM-L6-v2', batch: int=256, use_cache: bool=True, cache_root: Optional[Path]=None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError('sentence_transformers is not available in this environment') from e
        self._model_name = model
        self._st = SentenceTransformer(model)
        self.dim = int(self._st.get_sentence_embedding_dimension())
        self.batch = int(batch)
        self.name = f'sentence_transformers:{model}'
        self._cache = _DiskCache(f'st_{model}_{self.dim}', self.dim, root=cache_root) if use_cache else None

    def _compute(self, texts: list[str]) -> np.ndarray:
        vecs = self._st.encode(texts, batch_size=self.batch, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        return _cached_encode(texts, self.dim, self._cache, self._compute)

class TfidfEmbedder:

    def __init__(self, dim: int=384, seed: int=CFG.seed, use_cache: bool=False, cache_root: Optional[Path]=None):
        self.dim = int(dim)
        self.seed = int(seed)
        self.name = f'tfidf_svd:{self.dim}'
        self._fitted = False
        self._vectorizer = None
        self._svd = None
        self._svd_k = 0
        self._cache = _DiskCache(f'tfidf_{self.dim}_{self.seed}', self.dim, root=cache_root) if use_cache else None

    def _fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', strip_accents='unicode', min_df=1)
        try:
            X = self._vectorizer.fit_transform(texts)
        except ValueError:
            self._vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode', min_df=1, token_pattern='(?u)\\b\\w+\\b', vocabulary=['__empty__'])
            X = self._vectorizer.fit_transform(texts)
        n_docs, n_terms = X.shape
        if X.nnz == 0 or n_terms < 2:
            self._svd = None
            self._svd_k = 0
            self._fitted = True
            return
        k = min(self.dim, max(1, min(n_docs, n_terms) - 1), n_terms - 1)
        k = max(1, k)
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self._svd.fit(X)
        self._svd_k = k
        self._fitted = True

    def _project(self, texts: list[str]) -> np.ndarray:
        if self._svd is None or self._svd_k == 0:
            return np.zeros((len(texts), self.dim), dtype=np.float32)
        X = self._vectorizer.transform(texts)
        comp = self._svd.transform(X)
        n = comp.shape[0]
        out = np.zeros((n, self.dim), dtype=np.float32)
        out[:, :self._svd_k] = comp[:, :self._svd_k]
        return _l2_normalize(out)

    def fit(self, texts: list[str]) -> 'TfidfEmbedder':
        self._fit([t if t else ' ' for t in texts])
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not self._fitted:
            corpus = [t for t in texts if t]
            self._fit(corpus if corpus else [' '])

        def _compute(subset: list[str]) -> np.ndarray:
            return self._project(subset)
        return _cached_encode(texts, self.dim, self._cache, _compute)

def get_embedder(cfg=CFG):
    backend = (getattr(cfg, 'embed_backend', 'tfidf') or 'tfidf').lower()
    model = getattr(cfg, 'embed_model', 'all-minilm')
    dim = int(getattr(cfg, 'embed_dim', 384))
    if backend == 'ollama':
        if OllamaEmbedder.server_up():
            return OllamaEmbedder(model=model, dim=dim, cfg=cfg)
        return TfidfEmbedder(dim=dim)
    if backend == 'sentence_transformers':
        try:
            return SentenceTransformerEmbedder(model=model)
        except Exception:
            return TfidfEmbedder(dim=dim)
    return TfidfEmbedder(dim=dim)
__all__ = ['OllamaEmbedder', 'SentenceTransformerEmbedder', 'TfidfEmbedder', 'get_embedder']
