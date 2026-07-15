"""
Swappable LLM backend, same interface pattern as embeddings.py (one
"real" backend, one alternative) -- here, cloud API vs. Ollama, rather
than semantic vs. TF-IDF.

Why cloud API is the DEFAULT here specifically, unlike the papers RAG
project (which defaulted to Ollama): a cover letter and resume represent
YOU to a real employer. Text quality genuinely matters -- a small local
model tends to write noticeably weaker, more generic English/Korean than
a frontier model. Ollama stays available as a free, offline swap (good
for development/testing without burning API credits), one config change
away -- but it's not the default for the actual output you'd send out.

Neither concrete backend can be exercised end-to-end in this sandbox
(no API key here, no Ollama server running here) -- so both are built
against this interface and tested via a third, fully-offline
FakeLLMClient that proves the PROMPT CONSTRUCTION and GROUNDING logic
are correct, independent of which real backend eventually runs it. Same
principle as testing VectorStore logic with TfidfEmbedding while
SentenceTransformerEmbedding remains real-but-untested-here.
"""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class AnthropicLLMClient(LLMClient):
    """Real cloud backend. Requires ANTHROPIC_API_KEY set as an
    environment variable -- never hardcode a key in source."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OllamaLLMClient(LLMClient):
    """Free, local backend. Requires Ollama running locally
    (`ollama serve`, with a model already pulled, e.g. `ollama pull llama3`)."""

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


class FakeLLMClient(LLMClient):
    """Test double: records exactly what prompts it was called with, and
    returns a fixed canned response (or echoes the prompt back, useful
    for asserting on what content actually made it into the prompt).
    This is what proves letter_writer.py's grounding logic is correct,
    without needing a real API key or a running Ollama server."""

    def __init__(self, canned_response: str = "GENERATED TEXT"):
        self.canned_response = canned_response
        self.calls = []  # list of (system_prompt, user_prompt) tuples

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.canned_response
