import pytest
from django.test import override_settings

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
            query_str="q",
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
            query_str="q",
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
            query_str="q",
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
            query_str="q",
            allowed_ids=set(range(1, 30)),
            user=None,
        )

        assert result is not None
        assert len(result) == FUSED_TOP_DOCS

    def test_user_is_passed_to_fulltext_backend(self, mocker):
        backend = mocker.MagicMock()
        backend.search_ids.return_value = []
        mocker.patch("documents.search.get_backend", return_value=backend)
        user = object()

        hybrid_fused_document_ids(
            index=mocker.MagicMock(),
            query_str="q",
            allowed_ids={1},
            user=user,
        )

        assert backend.search_ids.call_args.args[1] is user


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
