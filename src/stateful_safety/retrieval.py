"""Deterministic chronological, BM25, dense and hybrid (RRF) history selection.

Every selector ranks ONLY the earlier steps it is given, then returns the chosen items in
chronological order for the judge. Ties are broken toward the more recent step.
"""
from __future__ import annotations
import math
import re
from collections import Counter

HISTORY_MODES = ("current", "recent", "full", "bm25", "dense", "hybrid")
RRF_K = 60  # standard reciprocal-rank-fusion constant (Cormack et al. 2009)


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.casefold())


def bm25_scores(query: str, prior: list[tuple[int, str]]) -> list[float]:
    """BM25 (k1=1.2, b=0.75) of each earlier step against the current request."""
    toks = [tokens(txt) for _, txt in prior]
    if not toks:
        return []
    df = Counter(word for doc in toks for word in set(doc))
    avdl = max(1., sum(len(doc) for doc in toks) / len(toks))
    qt = set(tokens(query))
    scores = []
    for doc in toks:
        freq = Counter(doc)
        value = 0.0
        for term in qt:
            n = df[term]
            idf = math.log(1 + (len(toks)-n+0.5)/(n+0.5))
            f = freq[term]
            if f:
                value += idf * ((2.2*f) / (f + 1.2*(0.25+0.75*len(doc)/avdl)))
        scores.append(value)
    return scores


def _ranked(prior: list[tuple[int, str]], scores: list[float]) -> list[int]:
    """Positions in prior, best first; ties go to the more recent step."""
    return sorted(range(len(prior)), key=lambda j: (-scores[j], -prior[j][0]))


def _top_k(prior: list[tuple[int, str]], order: list[int], k: int) -> list[tuple[int, str]]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    return sorted((prior[j] for j in order[:k]), key=lambda p: p[0])


def bm25_search(query: str, prior: list[tuple[int, str]], k: int) -> list[tuple[int, str]]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not prior or k == 0:
        return []
    return _top_k(prior, _ranked(prior, bm25_scores(query, prior)), k)


def recent_search(prior: list[tuple[int,str]], k: int) -> list[tuple[int,str]]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    return prior[-k:] if k else []


class DenseScorer:
    """General pretrained sentence encoder, loaded ONCE; each text is embedded once.

    No fine-tuning. Embeddings live only in memory for the run; never written to disk.
    """

    def __init__(self, model_name: str, revision: str | None = None, encoder=None):
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("Install project[retrieval] for dense retrieval") from e
            encoder = SentenceTransformer(model_name, revision=revision)
        self.encoder = encoder
        self.model_name = model_name
        self.revision = revision
        self._cache: dict[str, object] = {}
        self.texts_encoded = 0

    def _embed(self, texts: list[str]) -> list:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self.encoder.encode(missing, normalize_embeddings=True)
            self._cache.update(zip(missing, vectors))
            self.texts_encoded += len(missing)
        return [self._cache[t] for t in texts]

    def scores(self, query: str, prior: list[tuple[int, str]]) -> list[float]:
        if not prior:
            return []
        q, *docs = self._embed([query] + [text for _, text in prior])
        return [float(sum(a * b for a, b in zip(q, d))) for d in docs]


def dense_search(query: str, prior: list[tuple[int,str]], k: int, scorer: DenseScorer):
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not prior or k == 0:
        return []
    return _top_k(prior, _ranked(prior, scorer.scores(query, prior)), k)


def hybrid_search(query: str, prior: list[tuple[int,str]], k: int, scorer: DenseScorer):
    """Reciprocal-rank fusion of RELEVANCE-sorted BM25 and dense ranks, then chronological."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not prior or k == 0:
        return []
    fused = [0.0] * len(prior)
    for order in (_ranked(prior, bm25_scores(query, prior)),
                  _ranked(prior, scorer.scores(query, prior))):
        for rank, j in enumerate(order, start=1):
            fused[j] += 1.0 / (RRF_K + rank)
    return _top_k(prior, _ranked(prior, fused), k)


def select_history(mode: str, query: str, prior: list[tuple[int,str]], k: int,
                   dense: DenseScorer | str | None = None):
    if mode == "current":
        return []
    if mode == "full":
        return list(prior)
    if mode == "recent":
        return recent_search(prior,k)
    if mode == "bm25":
        return bm25_search(query,prior,k)
    if mode in ("dense", "hybrid"):
        if dense is None:
            raise ValueError("--dense-model required for dense/hybrid retrieval")
        scorer = dense if isinstance(dense, DenseScorer) else DenseScorer(dense)
        return (dense_search if mode == "dense" else hybrid_search)(query, prior, k, scorer)
    raise ValueError(f"Unsupported history mode: {mode}")
