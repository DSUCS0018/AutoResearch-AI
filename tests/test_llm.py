"""
tests/test_llm.py
-----------------
Unit tests for the Google Gemini LLM client (Stage 2).

All tests use mocking — no real API calls are made and no tokens are consumed.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.llm.client import GeminiClient, LLMConfigurationError, LLMResponseError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_response(text: str):
    """
    Build a mock object that mirrors the structure returned by
    client.models.generate_content(). The google-genai SDK exposes a
    top-level .text property on the response object.
    """
    response = MagicMock()
    response.text = text
    return response


# ── Test 1: Client initialization ─────────────────────────────────────────────

@patch("app.llm.client.genai")
def test_client_initialises_with_valid_key(mock_genai):
    """Client must initialise without error when an API key is present."""
    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "test-gemini-key-1234"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()

        # SDK Client constructor was called with the key
        mock_genai.Client.assert_called_once_with(api_key="test-gemini-key-1234")
        assert client._model == "gemini-2.5-flash-lite"


# ── Test 2: Missing API key ────────────────────────────────────────────────────

def test_missing_api_key_raises_configuration_error():
    """A missing GEMINI_API_KEY must raise LLMConfigurationError with a clear message."""
    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = ""
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        with pytest.raises(LLMConfigurationError) as exc_info:
            GeminiClient()

    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert ".env" in str(exc_info.value)


# ── Test 3: Successful generation ─────────────────────────────────────────────

@patch("app.llm.client.genai")
def test_generate_returns_text_on_success(mock_genai):
    """generate() must return the model's text on a successful API response."""
    expected = "RAG is a technique that combines retrieval with generation."

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = _make_mock_response(expected)
    mock_genai.Client.return_value = mock_client_instance

    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "test-key"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()
        result = client.generate("Explain RAG in one sentence.")

    assert result == expected
    mock_client_instance.models.generate_content.assert_called_once()


# ── Test 4: API / server error ────────────────────────────────────────────────

@patch("app.llm.client.genai")
def test_api_error_raises_llm_response_error(mock_genai):
    """A Gemini ServerError must be caught and re-raised as LLMResponseError."""
    from google.genai import errors as genai_errors

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = genai_errors.ServerError(
        500, {"message": "Internal error"}, None
    )
    mock_genai.Client.return_value = mock_client_instance

    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "test-key"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()

        with pytest.raises(LLMResponseError):
            client.generate("Hello")


# ── Test 5: Authentication / client error ─────────────────────────────────────

@patch("app.llm.client.genai")
def test_client_error_raises_configuration_error(mock_genai):
    """A Gemini ClientError (bad key / bad request) must raise LLMConfigurationError."""
    from google.genai import errors as genai_errors

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = genai_errors.ClientError(
        401, {"message": "API key not valid"}, None
    )
    mock_genai.Client.return_value = mock_client_instance

    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "bad-key"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()

        with pytest.raises(LLMConfigurationError):
            client.generate("Hello")


# ── Test 6: Empty response ─────────────────────────────────────────────────────

@patch("app.llm.client.genai")
def test_empty_response_raises_llm_response_error(mock_genai):
    """An empty model response must raise LLMResponseError."""
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = _make_mock_response("")
    mock_genai.Client.return_value = mock_client_instance

    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "test-key"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()

        with pytest.raises(LLMResponseError):
            client.generate("Hello")


# ── Test 7: Empty prompt validation ───────────────────────────────────────────

@patch("app.llm.client.genai")
def test_empty_prompt_raises_value_error(mock_genai):
    """Calling generate() with a blank prompt must raise ValueError immediately."""
    with patch("app.llm.client.config") as mock_cfg:
        mock_cfg.GEMINI_API_KEY = "test-key"
        mock_cfg.GEMINI_MODEL = "gemini-2.5-flash-lite"

        client = GeminiClient()

        with pytest.raises(ValueError):
            client.generate("   ")
