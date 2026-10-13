PY ?= python3

.PHONY: data ocr baseline analyze test demo claude lora lora-eval compare

data:        ## 400 clean invoices (+16 duplicates) with ground truth
	$(PY) -m invoiceai.cli generate --data data --n 400 --seed 7

ocr:         ## Tesseract over every image (cached in data/ocr)
	$(PY) -m invoiceai.cli ocr --data data

baseline:    ## rules-only extraction scored on the held-out test split
	$(PY) -m invoiceai.cli extract --backend rules --split test --out results/preds_rules.jsonl
	$(PY) -m invoiceai.cli evaluate --preds results/preds_rules.jsonl --name rules

analyze:     ## anomaly detection: oracle fields vs. end-to-end from OCR+rules
	$(PY) -m invoiceai.cli extract --backend rules --split all --out results/preds_rules_all.jsonl
	$(PY) -m invoiceai.cli analyze --oracle --name oracle
	$(PY) -m invoiceai.cli analyze --preds results/preds_rules_all.jsonl --name rules

test:
	$(PY) -m pytest -q

demo: data ocr baseline analyze

# ---- need an API key / GPU (see README) ----
claude:      ## prompted Claude (set ANTHROPIC_API_KEY; add --price-in/--price-out to the evaluate step)
	$(PY) -m invoiceai.cli extract --backend claude --model claude-haiku-5-5 --out results/preds_claude.jsonl
	$(PY) -m invoiceai.cli evaluate --preds results/preds_claude.jsonl --name claude-haiku-prompted

lora:        ## LoRA fine-tune a small open model on the train split
	$(PY) -m invoiceai.finetune_lora --data data --out models/lora-invoice

lora-eval:   ## score the base model (prompted) and the LoRA model on the same test split
	$(PY) -m invoiceai.cli extract --backend local --prompt full --out results/preds_qwen_base.jsonl
	$(PY) -m invoiceai.cli evaluate --preds results/preds_qwen_base.jsonl --name qwen0.5b-prompted
	$(PY) -m invoiceai.cli extract --backend local --adapter models/lora-invoice --out results/preds_qwen_lora.jsonl
	$(PY) -m invoiceai.cli evaluate --preds results/preds_qwen_lora.jsonl --name qwen0.5b-lora

compare:
	$(PY) -m invoiceai.cli compare
