from types import SimpleNamespace

from invoiceai.extract_llm import (ClaudeExtractor, build_prompt, normalize_prediction,
                                   parse_json_response)


def test_parse_plain_and_fenced_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('Here you go:\n```json\n{"a": {"b": 2}}\n```\nDone.') == {"a": {"b": 2}}


def test_parse_rejects_garbage_and_truncation():
    assert parse_json_response("no json here") is None
    assert parse_json_response('{"a": 1, "b":') is None
    assert parse_json_response("") is None


def test_normalize_prediction_coerces_types():
    out = normalize_prediction({
        "invoice_number": " ACH-2025-00001 ", "date": "03 Oct 2025", "vendor": "Atlas Cloud Hosting",
        "currency": "dh", "tax_amount": "320,36", "total_amount": 1922.16,
        "description": ["Object storage 5 TB", "Managed database plan"]})
    assert out["invoice_number"] == "ACH-2025-00001"
    assert out["date"] == "2025-10-03"
    assert out["currency"] == "MAD"
    assert out["tax_amount"] == 320.36 and out["total_amount"] == 1922.16
    assert out["description"] == "Object storage 5 TB; Managed database plan"


def test_normalize_handles_missing_and_none():
    assert normalize_prediction(None) is None
    out = normalize_prediction({"vendor": "", "total_amount": None})
    assert out["vendor"] is None and out["total_amount"] is None and out["date"] is None


def test_prompt_styles_embed_ocr_and_compact_is_shorter():
    ocr = "Atlas Cloud Hosting\nTotal 10.00 MAD"
    full, compact = build_prompt(ocr, "full"), build_prompt(ocr, "compact")
    assert ocr in full and ocr in compact
    assert len(compact) < len(full)


class _FakeMessages:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=321, output_tokens=45))


def _client(**kw):
    return SimpleNamespace(messages=_FakeMessages(**kw))


def test_claude_extractor_parses_response_and_records_usage():
    c = _client(text='```json\n{"vendor": "Atlas Cloud Hosting", "total_amount": "1,922.16", '
                     '"date": "2025-10-03", "currency": "MAD", "tax_amount": 320.36, '
                     '"invoice_number": "ACH-2025-61965", "description": null}\n```')
    res = ClaudeExtractor(model="m", client=c).extract("some ocr text")
    assert res.pred["total_amount"] == 1922.16 and res.pred["description"] is None
    assert (res.input_tokens, res.output_tokens) == (321, 45)
    assert res.latency_s >= 0 and not res.error
    call = c.messages.calls[0]
    assert call["model"] == "m" and "some ocr text" in call["messages"][0]["content"]
    assert "temperature" not in call


def test_claude_extractor_survives_api_errors():
    res = ClaudeExtractor(client=_client(error=RuntimeError("boom"))).extract("x")
    assert res.pred is None and "boom" in res.error


def test_claude_extractor_unparseable_output_gives_none():
    res = ClaudeExtractor(client=_client(text="Sorry, I cannot do that.")).extract("x")
    assert res.pred is None and res.raw.startswith("Sorry")
