import json
import urllib.error

from cogalpha.llm import OpenAICompatibleClient


def test_deepseek_env_builds_official_v4_pro_high_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "high")
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    monkeypatch.setenv("COGALPHA_LLM_MAX_TOKENS", "8192")
    monkeypatch.delenv("COGALPHA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_MODEL", raising=False)

    client = OpenAICompatibleClient.from_env()

    assert client.api_key == "secret"
    assert client.base_url == "https://api.deepseek.com"
    assert client.model == "deepseek-v4-pro"
    assert client.reasoning_effort == "high"
    assert client.thinking == "enabled"
    assert client.max_tokens == 8192
    assert client.response_format == "json_object"


def test_deepseek_thinking_payload_uses_reasoning_effort_and_omits_temperature(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        max_tokens=8192,
        reasoning_effort="high",
        thinking="enabled",
    )

    assert client.complete_json("Return JSON.", "TestSchema") == "{\"ok\": true}"

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "deepseek-v4-pro"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["reasoning_effort"] == "high"
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["max_tokens"] == 8192
    assert "temperature" not in captured["payload"]


def test_response_format_can_be_disabled(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        api_key="secret",
        base_url="https://example.invalid",
        model="custom-model",
        response_format=None,
    )

    client.complete_json("Return JSON.", "TestSchema")

    assert "response_format" not in captured["payload"]


def test_http_error_includes_provider_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs={},
            fp=_ErrorBody(b'{"error":"unsupported parameter"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        api_key="secret",
        base_url="https://example.invalid",
        model="custom-model",
    )

    try:
        client.complete_json("Return JSON.", "TestSchema")
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "unsupported parameter" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


class _ErrorBody:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        pass
