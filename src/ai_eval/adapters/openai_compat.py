"""Backwards-compatible import path; the implementation lives in chat_adapter."""

from ai_eval.adapters.chat_adapter import OpenAIAdapter as OpenAICompatAdapter

__all__ = ["OpenAICompatAdapter"]
