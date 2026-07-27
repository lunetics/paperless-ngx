"""
Hybrid retrieval for AI chat: fuse the dense vector ranking with the ranking
of the existing full-text search backend (``documents.search``).

Dense embeddings are weak for rare exact tokens (invoice numbers, dates,
quantities) that the full-text search resolves precisely; conversely, the
full-text ranking is weak for purely semantic questions. Weighted reciprocal
rank fusion combines both so that neither path can displace the other's
confident top results.

The fusion only ever *narrows* the candidate set the caller already resolved
(permissions stay intact) and falls back to ``None`` — meaning "leave the
existing retrieval untouched" — whenever the full-text side contributes
nothing or fails.
"""

import logging
from typing import TYPE_CHECKING

from paperless_ai.db import db_connection_released
from paperless_ai.indexing import _document_id_filters

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django.contrib.auth.models import AbstractUser
    from llama_index.core.indices import VectorStoreIndex

logger = logging.getLogger("paperless_ai.hybrid")

# Conservative defaults: the dense ranking is weighted double so a fused
# result can never displace a confident dense hit, and the full-text list is
# capped so broad natural-language matches do not flood the fusion.
DENSE_CANDIDATES = 40
FULLTEXT_CANDIDATES = 20
DENSE_WEIGHT = 2.0
FULLTEXT_WEIGHT = 1.0
RRF_K = 5
FUSED_TOP_DOCS = 5


def weighted_reciprocal_rank_fusion(
    rankings: "Sequence[tuple[Sequence[int], float]]",
    *,
    k: int = RRF_K,
) -> list[int]:
    """
    Fuse several document-id rankings into one.

    Each entry is ``(ranking, weight)``; a document's score is the weighted
    sum of ``weight / (k + position + 1)`` over all rankings it appears in.
    """
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position + 1)
    return sorted(scores, key=lambda doc_id: -scores[doc_id])


def _dense_document_ranking(
    index: "VectorStoreIndex",
    query_str: str,
    allowed_ids: set[int],
) -> list[int]:
    from llama_index.core.retrievers import VectorIndexRetriever

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DENSE_CANDIDATES,
        filters=_document_id_filters(str(doc_id) for doc_id in allowed_ids),
    )
    # Slow query-embedding + vector search; no ORM access during it. See
    # paperless_ai.db and #12976.
    with db_connection_released():
        nodes = retriever.retrieve(query_str)

    ranking: list[int] = []
    seen: set[int] = set()
    for node in nodes:
        try:
            doc_id = int(node.metadata["document_id"])
        except (KeyError, TypeError, ValueError):  # pragma: no cover
            continue
        if doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def hybrid_fused_document_ids(
    *,
    index: "VectorStoreIndex",
    query_str: str,
    allowed_ids: set[int],
    user: "AbstractUser | None",
) -> list[int] | None:
    """
    Return the fused top document ids for ``query_str``, or ``None`` when the
    caller should keep its existing (dense-only) retrieval unchanged.

    The full-text backend performs its own permission filtering via ``user``;
    intersecting with ``allowed_ids`` additionally guarantees the result can
    only narrow the caller's candidate set, never extend it.
    """
    # Imported lazily: this module is reached from documents.views via
    # paperless_ai.chat, importing documents.search at module load would be
    # circular.
    from documents.search import get_backend

    try:
        fulltext_ids = get_backend().search_ids(
            query_str,
            user,
            limit=FULLTEXT_CANDIDATES,
        )
    except Exception:
        logger.exception(
            "Full-text search failed during hybrid retrieval, "
            "falling back to dense-only retrieval.",
        )
        return None

    fulltext_ranking = [doc_id for doc_id in fulltext_ids if doc_id in allowed_ids]
    if not fulltext_ranking:
        # Nothing to fuse — keep today's behavior byte-identical.
        return None

    dense_ranking = _dense_document_ranking(index, query_str, allowed_ids)

    fused = weighted_reciprocal_rank_fusion(
        [
            (dense_ranking, DENSE_WEIGHT),
            (fulltext_ranking, FULLTEXT_WEIGHT),
        ],
    )
    return fused[:FUSED_TOP_DOCS] or None
