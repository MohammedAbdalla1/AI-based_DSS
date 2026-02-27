from types import SimpleNamespace

import pytest

from txt_to_sql import sql_generator as sg


@pytest.fixture(autouse=True)
def reset_client_cache():
    sg._client = None
    yield
    sg._client = None


def test_get_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(sg.SQLGenerationError, match="GEMINI_API_KEY is missing") as exc:
        sg._get_client()
    assert exc.value.subcode == "MISSING_API_KEY"


def test_get_client_caches_client(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            created.append(api_key)

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    first = sg._get_client()
    second = sg._get_client()

    assert first is second
    assert created == ["fake-key"]


def test_get_client_wraps_client_init_exception(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            raise RuntimeError("init failed")

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    with pytest.raises(
        sg.SQLGenerationError,
        match="Gemini client initialization failed: RuntimeError",
    ):
        sg._get_client()


@pytest.mark.parametrize("prompt", ["", "   ", None, 123])
def test_generate_sql_rejects_empty_prompt(prompt):
    with pytest.raises(sg.SQLGenerationError, match="Prompt is empty") as exc:
        sg.generate_sql(prompt)
    assert exc.value.subcode == "PROMPT_EMPTY"


def test_generate_sql_success(monkeypatch):
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="  SELECT id FROM users  ")

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    sql = sg.generate_sql("list all users")

    assert sql == "SELECT id FROM users"
    assert len(calls) == 1
    assert calls[0]["model"] == "models/gemini-2.5-flash"
    assert calls[0]["contents"] == "list all users"
    assert calls[0]["config"]["temperature"] == 0.1


def test_generate_sql_wraps_client_exception(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("boom")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    with pytest.raises(sg.SQLGenerationError, match="Gemini generation failed: RuntimeError") as exc:
        sg.generate_sql("list all users")
    assert exc.value.subcode == "PROVIDER_FAILURE"


@pytest.mark.parametrize(
    "model_text",
    ["", "   "],
)
def test_generate_sql_rejects_empty_response(monkeypatch, model_text):
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=model_text)

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    with pytest.raises(sg.SQLGenerationError, match="Empty response from LLM") as exc:
        sg.generate_sql("list all users")
    assert exc.value.subcode == "EMPTY_RESPONSE"


@pytest.mark.parametrize(
    "response_payload",
    [SimpleNamespace(text=None), SimpleNamespace(text=123), SimpleNamespace()],
)
def test_generate_sql_rejects_invalid_response_text_type(monkeypatch, response_payload):
    class FakeModels:
        def generate_content(self, **kwargs):
            return response_payload

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    with pytest.raises(sg.SQLGenerationError, match="Invalid response type from LLM") as exc:
        sg.generate_sql("list all users")
    assert exc.value.subcode == "INVALID_RESPONSE_TYPE"
