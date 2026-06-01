"""Provider-agnostic LLM interfaces."""

from cogalpha.llm.client import JSONCompletionClient, OpenAICompatibleClient

__all__ = ["JSONCompletionClient", "OpenAICompatibleClient"]
