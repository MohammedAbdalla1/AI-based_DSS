from types import SimpleNamespace

from llm_clients import ProviderCallError

from rag import generator as rg


def test_generate_answer_with_usage_falls_back_to_groq(monkeypatch):
    class FakeGeminiModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("boom")

        def embed_content(self, **kwargs):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

    class FakeGeminiClient:
        def __init__(self):
            self.models = FakeGeminiModels()

    monkeypatch.setattr(rg, "get_gemini_client", lambda: FakeGeminiClient())
    monkeypatch.setattr(rg, "get_settings", lambda: SimpleNamespace(rag_generation_model="gemini-model", rag_embedding_model="embedding-model"))
    monkeypatch.setattr(
        rg,
        "call_groq_completion",
        lambda *args, **kwargs: (
            "Groq answer.",
            SimpleNamespace(usage=SimpleNamespace(total_tokens=17)),
        ),
    )

    answer, tokens_used = rg.generate_answer_with_usage("What happened?", ["The revenue increased."])

    assert answer == "Groq answer."
    assert tokens_used == 17


def test_embed_texts_still_requires_gemini(monkeypatch):
    def _missing_key():
        raise ProviderCallError("gemini", "GEMINI_API_KEY is missing", "MISSING_API_KEY")

    monkeypatch.setattr(rg, "get_gemini_client", _missing_key)
    monkeypatch.setattr(rg, "get_settings", lambda: SimpleNamespace(rag_embedding_model="embedding-model"))

    try:
        rg.embed_texts(["hello world"])
    except rg.GeminiRAGError as exc:
        assert exc.subcode == "MISSING_API_KEY"
    else:
        raise AssertionError("Expected GeminiRAGError")
