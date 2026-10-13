"""LLM extractors: a prompted API model (Claude) and a local HF model (optionally LoRA-tuned).

Both take OCR text and return an ``ExtractionResult`` with the parsed fields plus
latency and token counts, which the evaluation uses for the cost/latency comparison.
The heavy imports (anthropic, torch, transformers, peft) are lazy so the rest of
the project and the tests run without them.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .schema import ALL_KEYS, AMOUNT_FIELDS, parse_amount, parse_currency, parse_date

FULL_PROMPT = """You extract structured data from the OCR text of a single invoice.

Return ONLY a JSON object with exactly these keys:
- "invoice_number": string, the invoice identifier printed by the vendor
- "date": invoice issue date as YYYY-MM-DD
- "vendor": name of the company that ISSUED the invoice (not the customer, "Bill to" or "Client")
- "currency": ISO 4217 code such as MAD or EUR (map "DH" to MAD and the euro sign to EUR)
- "tax_amount": VAT / TVA / tax amount as a number (dot as decimal separator, no thousands separator)
- "total_amount": final amount payable including tax, as a number
- "description": the line item descriptions joined with "; "

Use null for a field that is not present. The OCR text may contain small recognition errors.
Do not add any commentary.

OCR text:
\"\"\"
{ocr}
\"\"\"
"""

COMPACT_PROMPT = "Extract the invoice fields as JSON.\n\n{ocr}\n"

PROMPTS = {"full": FULL_PROMPT, "compact": COMPACT_PROMPT}


def build_prompt(ocr_text: str, style: str = "full") -> str:
    return PROMPTS[style].format(ocr=ocr_text.strip())


@dataclass
class ExtractionResult:
    pred: dict | None
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""
    error: str = ""


def parse_json_response(text: str) -> dict | None:
    """Pull the first JSON object out of a model response (handles code fences)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def normalize_prediction(d: dict | None) -> dict | None:
    """Coerce model output into the canonical types used by the evaluator."""
    if d is None:
        return None
    out: dict = {}
    for k in ALL_KEYS:
        v = d.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            out[k] = None
        elif k in AMOUNT_FIELDS:
            out[k] = parse_amount(v)
        elif k == "date":
            out[k] = parse_date(v) or str(v).strip()
        elif k == "currency":
            out[k] = parse_currency(v)
        elif isinstance(v, list):
            out[k] = "; ".join(str(x) for x in v)
        else:
            out[k] = str(v).strip()
    return out


class ClaudeExtractor:
    """Prompted extraction through the Anthropic Messages API."""

    def __init__(self, model: str = "claude-haiku-5-5", prompt: str = "full", client=None,
                 max_tokens: int = 400, temperature: float | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.prompt = client, model, prompt
        self.max_tokens, self.temperature = max_tokens, temperature

    def extract(self, ocr_text: str) -> ExtractionResult:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": build_prompt(ocr_text, self.prompt)}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:  # network / API error: record and move on
            return ExtractionResult(None, time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
        dt = time.perf_counter() - t0
        raw = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return ExtractionResult(
            pred=normalize_prediction(parse_json_response(raw)),
            latency_s=dt,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=raw,
        )


class LocalHFExtractor:
    """Greedy decoding with a local causal LM, optionally with a LoRA adapter merged in."""

    def __init__(self, base_model: str, adapter: str | None = None, prompt: str = "compact",
                 max_new_tokens: int = 300, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(adapter or base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        if self.device == "cuda":
            model = model.to(torch.bfloat16)
        self.model = model.to(self.device).eval()
        self.prompt, self.max_new_tokens = prompt, max_new_tokens

    def extract(self, ocr_text: str) -> ExtractionResult:
        torch = self.torch
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(ocr_text, self.prompt)}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0][enc["input_ids"].shape[1]:]
        raw = self.tok.decode(gen, skip_special_tokens=True)
        return ExtractionResult(
            pred=normalize_prediction(parse_json_response(raw)), latency_s=dt,
            input_tokens=int(enc["input_ids"].shape[1]), output_tokens=int(len(gen)), raw=raw)
