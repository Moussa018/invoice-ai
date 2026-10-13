"""LoRA fine-tuning of a small causal LM for OCR-text -> invoice JSON.

Needs the optional dependencies in ``requirements-llm.txt`` (torch, transformers,
peft). A GPU is recommended; a few hundred short examples also train on CPU with
a ~0.5B model, just slowly.

    python -m invoiceai.finetune_lora --data data --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --out models/lora-invoice
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import load_ocr_text, load_truth
from .extract_llm import build_prompt

TARGET_KEYS = ["invoice_number", "date", "vendor", "currency", "tax_amount", "total_amount", "description"]


def target_json(truth: dict) -> str:
    obj = {k: truth[k] for k in TARGET_KEYS}
    obj["tax_amount"] = round(float(obj["tax_amount"]), 2)
    obj["total_amount"] = round(float(obj["total_amount"]), 2)
    return json.dumps(obj, ensure_ascii=False)


def build_sft_examples(data_dir: str | Path, split: str = "train", prompt_style: str = "compact") -> list[dict]:
    """(prompt, completion) pairs from OCR text and ground truth. No torch needed."""
    examples = []
    for r in load_truth(data_dir, split):
        ocr = load_ocr_text(data_dir, r["id"])
        if not ocr.strip():
            continue
        examples.append({"id": r["id"], "prompt": build_prompt(ocr, prompt_style),
                         "completion": target_json(r)})
    return examples


def tokenize_example(tok, ex: dict, max_len: int) -> dict:
    """Mask the prompt so the loss is computed on the JSON completion only."""
    prompt_text = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                          tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(ex["completion"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def train(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = build_sft_examples(args.data, "train", args.prompt)
    if not examples:
        raise SystemExit("no training examples: run `generate` and `ocr` first")
    feats = [tokenize_example(tok, ex, args.max_len) for ex in examples]

    class ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(feats)

        def __getitem__(self, i):
            return feats[i]

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(torch.bfloat16)
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable, total = model.get_nb_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_trainer"), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10,
        save_strategy="no", report_to="none", bf16=cuda, remove_unused_columns=False,
        seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0

    out = Path(args.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({
        "base_model": args.base_model, "prompt_style": args.prompt, "lora_r": args.r,
        "epochs": args.epochs, "lr": args.lr, "n_examples": len(examples),
        "trainable_params": trainable, "total_params": total,
        "train_seconds": round(seconds, 1), "device": "cuda" if cuda else "cpu",
    }, indent=2))
    print(f"saved adapter to {out} ({seconds:.0f}s, {trainable:,} trainable params)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="models/lora-invoice")
    p.add_argument("--prompt", default="compact", choices=["compact", "full"])
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
