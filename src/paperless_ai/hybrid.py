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

Why not llama-index's ``QueryFusionRetriever``: it fuses ``NodeWithScore``
lists produced by ``BaseRetriever`` instances, while the full-text backend
yields bare document ids (and the consumer needs a document-id set for a
``MetadataFilters`` IN filter); wrapping ids in a synthetic retriever would
fabricate nodes, and its default ``num_queries`` triggers LLM query
generation per turn. Document-level weighted RRF over ids is the smaller
mechanism.
"""

import logging
from typing import TYPE_CHECKING

from paperless_ai.db import db_connection_released
from paperless_ai.indexing import _document_id_filters

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from django.contrib.auth.models import AbstractUser
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.schema import QueryBundle

logger = logging.getLogger("paperless_ai.hybrid")

# Conservative defaults: the dense ranking is weighted double so a document
# found ONLY by full-text cannot displace the leading dense ranks on its
# full-text position alone (it can still overtake far-tail dense ranks).
# Documents present in BOTH rankings accumulate combined scores and may
# re-order — and thereby displace — dense-only results from the fused top
# slots; that re-ranking is the point of the fusion. The full-text list is
# capped so broad natural-language matches do not flood the fusion.
# Unit caveat: DENSE_CANDIDATE_NODES bounds retrieved CHUNKS (llama-index's
# similarity_top_k operates on nodes), while FULLTEXT_CANDIDATES bounds
# DOCUMENTS. The dense chunk window is a hard gate — a document without a
# chunk in it cannot be recovered by fusion — so on corpora of short
# (single-chunk) documents it spans ~40 documents, on long-document corpora
# far fewer.
DENSE_CANDIDATE_NODES = 40
FULLTEXT_CANDIDATES = 20
DENSE_WEIGHT = 2.0
FULLTEXT_WEIGHT = 1.0
RRF_K = 5
# Kept below CHAT_RETRIEVER_TOP_K on purpose: the coverage pass grants each
# fused document one chunk slot first, so with fewer fused documents than
# total slots the dense leaders keep their additional chunks. The value 3
# comes from an external 12-question production-corpus evaluation (every
# full-text winner ranked <= 3 there — not reproducible from this repo);
# revalidate when tuning: larger values trade dense context depth for more
# full-text candidates.
FUSED_TOP_DOCS = 3


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
    query_bundle: "QueryBundle",
    allowed_ids: set[int],
) -> list[int]:
    from llama_index.core.retrievers import VectorIndexRetriever

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DENSE_CANDIDATE_NODES,
        filters=_document_id_filters(str(doc_id) for doc_id in allowed_ids),
    )
    # Slow query-embedding + vector search; no ORM access during it. See
    # paperless_ai.db and #12976. Retrieving with the shared QueryBundle
    # lets llama-index cache the computed embedding on the bundle, so the
    # caller's subsequent retrieval reuses it instead of re-embedding
    # (verified against llama-index-core 0.14.22: the sync retrieve path
    # mutates the passed bundle; the async path copies — chat is sync).
    with db_connection_released():
        nodes = retriever.retrieve(query_bundle)

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
    query_bundle: "QueryBundle",
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
    if len(allowed_ids) <= 1:
        # Fusion cannot narrow or usefully re-rank a single-document
        # candidate set — skip the extra full-text search and dense ranking
        # entirely (the single-document chat path is the most
        # latency-sensitive one).
        return None

    # Imported lazily: this module is reached from documents.views via
    # paperless_ai.chat, importing documents.search at module load would be
    # circular.
    from documents.search import get_backend

    try:
        fulltext_ids = get_backend().search_ids(
            query_bundle.query_str,
            # search_ids applies no permission filter only for user=None;
            # mirror the superuser mapping every other call site uses
            # (documents/views.py), otherwise admins get an over-restrictive
            # owner/shared-only full-text side.
            None if user is not None and user.is_superuser else user,
            # Deliberately the default QUERY mode: TEXT mode builds a strict
            # consecutive-token phrase query, which matches near nothing for
            # multi-word conversational questions (measured: the motivating
            # exact-token cases return zero results in TEXT mode). QUERY mode
            # keeps full recall; the ~1-in-5 conversational inputs its
            # structured parser rejects are handled by the ValueError
            # fallback below.
            limit=FULLTEXT_CANDIDATES,
        )
    except ValueError:
        # Parse-level failures (SearchQueryError is a ValueError) caused by
        # unusual user input are expected — fall back to dense-only
        # retrieval without paging operators.
        logger.debug(
            "Full-text query parse failed during hybrid retrieval, "
            "falling back to dense-only retrieval.",
            exc_info=True,
        )
        return None
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

    dense_ranking = _dense_document_ranking(index, query_bundle, allowed_ids)

    fused = weighted_reciprocal_rank_fusion(
        [
            (dense_ranking, DENSE_WEIGHT),
            (fulltext_ranking, FULLTEXT_WEIGHT),
        ],
    )
    return fused[:FUSED_TOP_DOCS] or None


def _node_document_id(node: "NodeWithScore") -> int | None:
    try:
        return int(node.metadata["document_id"])
    except (KeyError, TypeError, ValueError):  # pragma: no cover
        return None


def ensure_document_coverage(
    nodes: "Sequence[NodeWithScore]",
    document_ids: "Iterable[int]",
    limit: int,
) -> "list[NodeWithScore]":
    """
    Select up to ``limit`` nodes so that every document in ``document_ids``
    that has a node at all contributes its highest-ranked one (first in
    retriever order, which the vector store returns by descending
    similarity) — in fused order, ahead of the remaining nodes in retriever
    order. Without this, a plain top-k node cut has no per-document floor
    and can silently drop a fused winner whose chunks score below the other
    candidates' chunks; leading with fused order also keeps the reference
    list consistent with the fusion ranking.
    """
    best_by_doc: dict[int, NodeWithScore] = {}
    for node in nodes:
        doc_id = _node_document_id(node)
        if doc_id is not None and doc_id not in best_by_doc:
            best_by_doc[doc_id] = node

    result: list[NodeWithScore] = []
    picked_ids: set[int] = set()
    for doc_id in document_ids:
        if len(result) >= limit:
            break
        node = best_by_doc.get(doc_id)
        if node is not None and id(node) not in picked_ids:
            result.append(node)
            picked_ids.add(id(node))
    for node in nodes:
        if len(result) >= limit:
            break
        if id(node) not in picked_ids:
            result.append(node)
            picked_ids.add(id(node))
    return result
