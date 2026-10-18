"""
app/llm/client.py
-----------------
Reusable Google Gemini API client for AutoResearch AI.

All Gemini SDK interactions are isolated here. The rest of the application
imports this module and calls generate() without knowing SDK details.

Usage:
    from app.llm.client import GeminiClient

    client = GeminiClient()
    text = client.generate("Explain RAG in one sentence.")
"""

import logging
from google import genai
from google.genai import errors as genai_errors
from app.config import config

logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Raised when the LLM client cannot be configured (e.g. missing key)."""


class LLMResponseError(Exception):
    """Raised when the LLM API call fails or returns an unusable response."""


class GeminiClient:
    """
    Thin wrapper around the Google GenAI Python SDK.

    Responsibilities:
      - Validate configuration on construction.
      - Provide a single generate() method for text generation.
      - Translate SDK errors into clean application-level errors.
      - Log request lifecycle without exposing secrets.
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini client.

        Raises:
            LLMConfigurationError: If GEMINI_API_KEY is not configured.
        """
        if not config.GEMINI_API_KEY:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured. "
                "Please add it to your .env file."
            )

        # Pass the key explicitly so we never rely on ambient env state.
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._model = config.GEMINI_MODEL
        logger.debug("GeminiClient initialised (model=%s)", self._model)

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured Gemini model and return the response.

        Args:
            prompt: The user prompt to send to the model.

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMConfigurationError: If authentication / permission fails.
            LLMResponseError:      On API errors or empty replies.
            ValueError:            If prompt is blank.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        logger.info("Sending request to Gemini (model=%s)...", self._model)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except genai_errors.ClientError as exc:
            # 400/401/403 class errors — likely bad key or invalid request
            raise LLMConfigurationError(
                "Gemini authentication or request error. "
                "Verify that GEMINI_API_KEY in your .env file is correct. "
                f"Details: {exc}"
            ) from exc
        except genai_errors.ServerError as exc:
            raise LLMResponseError(
                f"Gemini server error: {exc}"
            ) from exc
        except Exception as exc:
            # Catch-all for unexpected SDK errors without leaking internals
            raise LLMResponseError(
                f"Unexpected error contacting Gemini API: {type(exc).__name__}"
            ) from exc

        text = self._extract_text(response)

        if not text:
            raise LLMResponseError(
                "Gemini returned an empty response. "
                "Try rephrasing the prompt."
            )

        logger.info("Gemini response received successfully.")
        return text

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(response) -> str:
        """
        Pull plain text from a Gemini generate_content() response.

        The google-genai SDK provides a top-level .text property that
        aggregates all text parts from the first candidate. We use that
        directly and fall back to manual traversal for robustness.
        """
        # Primary: response.text is the SDK's built-in convenience accessor
        try:
            if response.text:
                return response.text.strip()
        except (AttributeError, ValueError):
            pass

        # Fallback: walk candidates → content → parts
        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text.strip()
        except (AttributeError, TypeError):
            pass

        return ""
