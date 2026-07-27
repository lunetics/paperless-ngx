from __future__ import annotations

from unittest import mock

from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase


class TestChatStreamingViewUserForwarding(APITestCase):
    def test_view_passes_request_user_to_chat(self) -> None:
        """The view must hand request.user to the chat layer — the full-text
        side of hybrid retrieval derives its permission filter from it."""
        from django.contrib.auth.models import Permission

        user = User.objects.create_user(username="chat_user")
        user.user_permissions.add(
            Permission.objects.get(codename="view_document"),
        )
        self.client.force_authenticate(user=user)

        ai_config = mock.MagicMock()
        ai_config.ai_enabled = True
        with (
            mock.patch("documents.views.AIConfig", return_value=ai_config),
            mock.patch(
                "documents.views.stream_chat_with_documents",
                return_value=iter(["ok"]),
            ) as chat_mock,
        ):
            resp = self.client.post(
                "/api/documents/chat/",
                {"q": "a question"},
                format="json",
            )

        assert resp.status_code == status.HTTP_200_OK
        assert chat_mock.call_args.kwargs["user"] == user


class TestChatStreamingViewInputValidation(APITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.user = User.objects.create_superuser(username="temp_admin")
        self.client.force_authenticate(user=self.user)

    def _mock_ai_enabled(self) -> mock.MagicMock:
        """Return a mock AIConfig instance with ai_enabled=True."""
        m = mock.MagicMock()
        m.ai_enabled = True
        return m

    def test_oversized_question_is_rejected(self) -> None:
        with mock.patch(
            "documents.views.AIConfig",
            return_value=self._mock_ai_enabled(),
        ):
            resp = self.client.post(
                "/api/documents/chat/",
                {"q": "x" * 4001},
                format="json",
            )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_missing_question_is_rejected(self) -> None:
        with mock.patch(
            "documents.views.AIConfig",
            return_value=self._mock_ai_enabled(),
        ):
            resp = self.client.post(
                "/api/documents/chat/",
                {},
                format="json",
            )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
