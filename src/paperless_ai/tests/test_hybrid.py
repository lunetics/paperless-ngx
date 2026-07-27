from types import SimpleNamespace

import pytest
from django.test import override_settings
from llama_index.core.schema import QueryBundle

from documents.tests.factories import DocumentFactory
from paperless.config import AIConfig
from paperless.models import ApplicationConfiguration
from paperless_ai import chat
from paperless_ai.hybrid import FUSED_TOP_DOCS
from paperless_ai.hybrid import hybrid_fused_document_ids
from paperless_ai.hybrid import weighted_reciprocal_rank_fusion


class TestWeightedReciprocalRankFusion:
    def test_single_ranking_preserves_order(self):
        fused = weighted_reciprocal_rank_fusion([([1, 2, 3], 1.0)])
        assert fused == [1, 2, 3]

    def test_equal_weights_interleave_by_rank(self):
        fused = weighted_reciprocal_rank_fusion([([1, 2], 1.0), ([3, 4], 1.0)])
        # Same positions, same weights: first-ranked items of both lists
        # outscore both second-ranked items.
        assert set(fused[:2]) == {1, 3}
        assert set(fused[2:]) == {2, 4}

    def test_document_in_both_rankings_outranks_single_list_presence(self):
        fused = weighted_reciprocal_rank_fusion(
            [([1, 2], 1.0), ([2, 3], 1.0)],
        )
        assert fused[0] == 2

    def test_higher_weight_dominates_equal_ranks(self):
        fused = weighted_reciprocal_rank_fusion(
            [([1], 2.0), ([2], 1.0)],
        )
        assert fused == [1, 2]

    def test_empty_rankings_yield_empty_result(self):
        assert weighted_reciprocal_rank_fusion([]) == []
        assert weighted_reciprocal_rank_fusion([([], 1.0)]) == []

    def test_double_presence_may_displace_dense_only_hits(self):
        """Pins the documented displacement semantics: a document ranked in
        BOTH lists accumulates combined scores and may push dense-only
        documents out of the fused top slots; a full-text-only document
        cannot outrank the dense leader on its own."""
        fused = weighted_reciprocal_rank_fusion(
            [(list(range(1, 11)), 2.0), ([6, 7, 8, 999], 1.0)],
        )
        assert fused[:5] == [6, 1, 7, 2, 8]
        # Dense-only documents 3, 4, 5 are displaced from the top five.
        assert {3, 4, 5}.isdisjoint(fused[:5])
        # The full-text-only document participates but cannot reach the
        # leading slots on its full-text position alone.
        assert 999 in fused
        assert 999 not in fused[:5]


class TestHybridFusedDocumentIds:
    def test_empty_fulltext_result_returns_none_without_dense_query(
        self,
        mocker,
    ):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)
        dense = mocker.patch("paperless_ai.hybrid._dense_document_ranking")

        result = hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={1, 2},
            user=None,
        )

        assert result is None
        dense.assert_not_called()

    def test_fulltext_failure_falls_back_to_none(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.side_effect = RuntimeError("index unavailable")
        mocker.patch("documents.search.get_backend", return_value=backend)

        result = hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={1},
            user=None,
        )

        assert result is None

    def test_fulltext_ids_outside_allowed_set_are_dropped(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = [99, 1]
        mocker.patch("documents.search.get_backend", return_value=backend)
        mocker.patch(
            "paperless_ai.hybrid._dense_document_ranking",
            return_value=[2],
        )

        result = hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={1, 2},
            user=None,
        )

        assert result is not None
        assert 99 not in result
        assert set(result) == {1, 2}

    def test_result_is_capped_to_fused_top_docs(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = list(range(1, 30))
        mocker.patch("documents.search.get_backend", return_value=backend)
        mocker.patch(
            "paperless_ai.hybrid._dense_document_ranking",
            return_value=list(range(1, 30)),
        )

        result = hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids=set(range(1, 30)),
            user=None,
        )

        assert result is not None
        assert len(result) == FUSED_TOP_DOCS

    def test_single_document_set_skips_hybrid_entirely(self, mocker):
        get_backend = mocker.patch("documents.search.get_backend")

        result = hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={42},
            user=None,
        )

        assert result is None
        get_backend.assert_not_called()

    def test_regular_user_is_passed_to_fulltext_backend(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)
        user = mocker.MagicMock(is_superuser=False)

        hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={1, 2},
            user=user,
        )

        assert backend.search_ids.call_args.args[1] is user

    def test_superuser_is_mapped_to_none_like_sibling_call_sites(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)
        user = mocker.MagicMock(is_superuser=True)

        hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("q"),
            allowed_ids={1, 2},
            user=user,
        )

        assert backend.search_ids.call_args.args[1] is None

    def test_fulltext_keeps_default_query_mode_for_recall(self, mocker):
        """TEXT mode builds a strict consecutive-token phrase query that
        matches near nothing for conversational questions — the default
        QUERY mode keeps recall; its parse failures are handled by the
        ValueError fallback."""
        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)

        hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_bundle=QueryBundle("Frage: wie hoch war die Stromrechnung?"),
            allowed_ids={1, 2},
            user=None,
        )

        assert "search_mode" not in backend.search_ids.call_args.kwargs

    def test_parse_error_falls_back_without_error_log(self, mocker, caplog):
        backend = mocker.MagicMock()
        backend.search_ids.side_effect = ValueError("Field does not exist: 'Frage'")
        mocker.patch("documents.search.get_backend", return_value=backend)

        import logging

        with caplog.at_level(logging.DEBUG, logger="paperless_ai.hybrid"):
            result = hybrid_fused_document_ids(
                index=mocker.MagicMock(),
                query_bundle=QueryBundle("Frage: wie hoch war die Stromrechnung?"),
                allowed_ids={1, 2},
                user=None,
            )

        assert result is None
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def _fake_node(document_id: int) -> SimpleNamespace:
    return SimpleNamespace(metadata={"document_id": str(document_id)})


class TestEnsureDocumentCoverage:
    def test_low_scoring_fused_winner_gets_a_slot(self):
        from paperless_ai.hybrid import ensure_document_coverage

        # Score order: docs 1,1,2,2,3 then the full-text winner 9 at the end.
        nodes = [
            _fake_node(1),
            _fake_node(1),
            _fake_node(2),
            _fake_node(2),
            _fake_node(3),
            _fake_node(9),
        ]
        picked = ensure_document_coverage(nodes, [9, 1, 2, 3], limit=5)

        assert len(picked) == 5
        assert {n.metadata["document_id"] for n in picked} >= {"1", "2", "3", "9"}
        # Fused order leads: the full-text winner heads the selection.
        assert picked[0].metadata["document_id"] == "9"

    def test_fused_document_without_nodes_is_skipped(self):
        from paperless_ai.hybrid import ensure_document_coverage

        nodes = [_fake_node(1), _fake_node(2)]
        picked = ensure_document_coverage(nodes, [7, 1], limit=5)

        assert [n.metadata["document_id"] for n in picked] == ["1", "2"]

    def test_limit_is_respected(self):
        from paperless_ai.hybrid import ensure_document_coverage

        nodes = [_fake_node(i) for i in range(1, 8)]
        picked = ensure_document_coverage(nodes, [5, 6], limit=3)

        assert len(picked) == 3
        assert {"5", "6"} <= {n.metadata["document_id"] for n in picked}


@pytest.mark.django_db
class TestAIConfigHybridFlag:
    def test_defaults_to_false(self):
        assert AIConfig().llm_hybrid_retrieval is False

    @override_settings(LLM_HYBRID_RETRIEVAL=True)
    def test_settings_enable_flag(self):
        assert AIConfig().llm_hybrid_retrieval is True

    def test_app_config_enables_flag(self):
        app_config = ApplicationConfiguration.objects.first()
        app_config.llm_hybrid_retrieval = True
        app_config.save()
        assert AIConfig().llm_hybrid_retrieval is True


@pytest.mark.django_db
class TestChatHybridIntegration:
    def _capture_retriever_filters(self, mocker):
        captured_filters = []
        mock_retriever = mocker.MagicMock()
        mock_retriever.retrieve.return_value = []

        def capture(*args, **kwargs):
            captured_filters.append(kwargs.get("filters"))
            return mock_retriever

        mocker.patch("paperless_ai.chat.AIClient")
        # VectorIndexRetriever is imported lazily inside the functions under
        # test; patch it at the llama_index source.
        mocker.patch(
            "llama_index.core.retrievers.VectorIndexRetriever",
            side_effect=capture,
        )
        return captured_filters

    def test_flag_disabled_never_calls_fulltext_backend(
        self,
        temp_llm_index_dir,
        mock_embed_model,
        mocker,
    ):
        document = DocumentFactory.create(content="some content")
        self._capture_retriever_filters(mocker)
        get_backend = mocker.patch("documents.search.get_backend")

        list(chat.stream_chat_with_documents("question?", [document]))

        get_backend.assert_not_called()

    @override_settings(LLM_HYBRID_RETRIEVAL=True)
    def test_fused_ids_narrow_the_retriever_filter(
        self,
        temp_llm_index_dir,
        mock_embed_model,
        mocker,
    ):
        included = DocumentFactory.create(content="included document content")
        sibling = DocumentFactory.create(content="sibling document content")
        captured_filters = self._capture_retriever_filters(mocker)

        backend = mocker.MagicMock()
        backend.search_ids.return_value = [included.pk]
        mocker.patch("documents.search.get_backend", return_value=backend)

        list(
            chat.stream_chat_with_documents(
                "question?",
                [included, sibling],
            ),
        )

        # Two retriever constructions: the hybrid dense ranking (scoped to all
        # allowed documents) and the main retrieval (scoped to the fused set).
        assert len(captured_filters) == 2
        dense_values = captured_filters[0].filters[0].value
        assert set(dense_values) == {str(included.pk), str(sibling.pk)}
        fused_values = captured_filters[1].filters[0].value
        assert fused_values == [str(included.pk)]

    @override_settings(LLM_HYBRID_RETRIEVAL=True)
    def test_empty_fulltext_keeps_filter_identical_to_today(
        self,
        temp_llm_index_dir,
        mock_embed_model,
        mocker,
    ):
        included = DocumentFactory.create(content="included document content")
        sibling = DocumentFactory.create(content="sibling document content")
        captured_filters = self._capture_retriever_filters(mocker)

        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)

        list(
            chat.stream_chat_with_documents(
                "question?",
                [included, sibling],
            ),
        )

        # Fallback: only the main retriever runs, with the unchanged filter.
        assert len(captured_filters) == 1
        filter_values = captured_filters[0].filters[0].value
        assert set(filter_values) == {str(included.pk), str(sibling.pk)}
