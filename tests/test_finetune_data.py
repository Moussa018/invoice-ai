import json

from invoiceai.data import ocr_path
from invoiceai.finetune_lora import build_sft_examples, target_json, tokenize_example


class FakeTok:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_target_json_is_valid_and_rounded(small_data):
    _, records = small_data
    obj = json.loads(target_json(records[0]))
    assert set(obj) == {"invoice_number", "date", "vendor", "currency", "tax_amount",
                        "total_amount", "description"}
    assert obj["total_amount"] == records[0]["total_amount"]


def test_sft_examples_only_use_the_requested_split_and_need_ocr(fresh_data):
    out, records = fresh_data
    train_ids = [r["id"] for r in records if r["split"] == "train"]
    assert build_sft_examples(out, "train") == []          # no OCR text yet
    for i in train_ids[:3]:
        ocr_path(out, i).parent.mkdir(exist_ok=True)
        ocr_path(out, i).write_text(f"fake ocr for {i}", encoding="utf-8")
    ex = build_sft_examples(out, "train")
    assert [e["id"] for e in ex] == train_ids[:3]
    assert "fake ocr for" in ex[0]["prompt"]
    assert json.loads(ex[0]["completion"])["vendor"]
    assert build_sft_examples(out, "test") == []


def test_prompt_tokens_are_masked_from_the_loss():
    ex = {"prompt": "hello", "completion": '{"a": 1}'}
    feat = tokenize_example(FakeTok(), ex, max_len=1000)
    n_prompt = len("<user>hello<assistant>")
    assert feat["labels"][:n_prompt] == [-100] * n_prompt
    assert feat["labels"][n_prompt:] == feat["input_ids"][n_prompt:]
    assert len(feat["labels"]) == len(feat["input_ids"]) == len(feat["attention_mask"])
    assert feat["input_ids"][-5:] == [ord(c) for c in "<eos>"]


def test_truncation_applies_to_ids_and_labels_equally():
    feat = tokenize_example(FakeTok(), {"prompt": "x" * 50, "completion": "y" * 50}, max_len=40)
    assert len(feat["input_ids"]) == len(feat["labels"]) == 40
