# Invoice extraction and anomaly detection

Read invoice images, extract the key fields, then flag the invoices a cost-optimization
reviewer would want to look at: **duplicates**, **odd amounts** and **misclassified cost categories**.

The project also sets up a fair comparison between a **prompted LLM** and a **small LoRA fine-tuned
model** on accuracy, latency and cost. Everything except those two model runs has been executed
and measured; those two need an API key / a GPU and are marked clearly below.

```
invoice PNG ──► Tesseract OCR ──► extractor ──► structured fields ──► anomaly detectors ──► review list
                                   │                                   ├─ duplicates (rules)
                                   ├─ rules (regex baseline)           ├─ odd amounts (IsolationForest + robust z)
                                   ├─ prompted Claude                  └─ misclassified costs (TF-IDF + LogReg)
                                   └─ small open LLM (+ LoRA)                     ▲
                                                                       ledger.csv (booked category)
```

## Quickstart

```bash
sudo apt install tesseract-ocr          # macOS: brew install tesseract
pip install -r requirements.txt

make demo        # generate 416 invoices, OCR them (~5 min on 2 CPUs), run baseline + anomaly detection
make test        # 60 tests, ~20 s
```

Individual steps (`python -m invoiceai.cli --help`): `generate`, `ocr`, `extract`, `evaluate`,
`analyze`, `compare`.

## Layout

| Path | What it does |
|---|---|
| `invoiceai/generate.py` | Synthetic invoices (3 layouts, EN/FR labels, 3 number and date formats, scan noise) + ground truth + injected anomalies |
| `invoiceai/ocr.py` | Tesseract wrapper, results cached in `data/ocr/` |
| `invoiceai/extract_rules.py` | Regex baseline |
| `invoiceai/extract_llm.py` | `ClaudeExtractor` (Messages API) and `LocalHFExtractor` (any HF causal LM, optional LoRA adapter); both record latency and token counts |
| `invoiceai/finetune_lora.py` | LoRA SFT on OCR text → JSON, loss on the completion only |
| `invoiceai/anomalies.py` | The three detectors |
| `invoiceai/evaluate.py` | Field-level accuracy, latency percentiles, cost, anomaly precision/recall |
| `tests/` | 60 tests, including real Tesseract OCR on generated images and stubbed-model CLI round trips |

## Data

Real invoices are private, so the project generates its own. Each invoice has a vendor, number,
date, 1–3 line items, VAT and a total, rendered as a slightly rotated, blurred, grainy PNG.
Three vendor-specific layouts use different labels (`Invoice No` / `Facture N°` / `Invoice #`,
`VAT` / `TVA` / `Tax`, `Total` / `Total TTC` / `Amount due`), currencies (`MAD`, `DH`, `EUR`, `€`),
number formats (`1,234.56` / `1 234,56` / `1234.56`) and date formats. A "Bill to" line keeps the
customer name on the page as a distractor for the vendor field.

Three files come out next to the images:

- `truth.jsonl` – ground truth for the six scored fields (+ description, split)
- `ledger.csv` – the category each invoice was *booked* under (an input to the misclassification check)
- `labels.csv` – injected anomaly labels, used only for evaluation

Injected anomalies (default 400 base invoices + 16 duplicates): 16 **duplicates** (60 % exact copies,
40 % re-keyed with a changed invoice number), 16 **amount outliers** (3.5–12× the vendor's typical
amount), 20 **misclassified** costs (booked under a wrong category). The three types never overlap.
Duplicates inherit their source's train/test split so nothing leaks. The default split is 232 train /
184 test.

## Methods

**Extraction.** The six scored fields are `invoice_number`, `date`, `vendor`, `currency`,
`tax_amount`, `total_amount`. A seventh, `description` (line items joined with `; `), feeds the
category check and is scored with token-F1. A prediction counts as correct if it matches after
normalisation (case/accents/punctuation for names, ±0.005 for amounts, ISO dates).

**Anomalies.**
- *Duplicates*: same (fuzzily matched) vendor and same invoice number, **or** same vendor, same total
  (±0.01) and a date within 3 days. The later invoice is flagged and points at the earlier one.
- *Odd amounts*: per-vendor robust z-score of log(total) (median/MAD) plus a tax-ratio consistency
  feature go into an IsolationForest; an invoice is flagged only if the forest calls it an outlier
  **and** `|z| ≥ 3` or the tax ratio is off. The forest alone over-flags the natural tail; the z
  guard alone ignores arithmetic mismatches.
- *Misclassified costs*: TF-IDF (word + char n-grams) of vendor + description → logistic regression,
  trained on the ledger itself with **out-of-fold** predictions so an invoice never votes on its own
  label. Flagged when the model confidently disagrees with the booked category.

## Results (measured in this repo)

OCR runs at roughly 0.9 s per image on one CPU thread; the regex extractor is effectively instant.

### Extraction – rules baseline (no model)

| Data | docs | invoice no. | date | vendor | currency | tax | total | macro | all 6 correct | description F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Clean set, test split | 184 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 99.5 % | 100.0 % | **99.9 %** | 99.5 % | 0.94 |
| Noisy set (scan noise ×2.2, seed 21) | 156 | 91.0 % | 98.1 % | 97.4 % | 97.4 % | 92.3 % | 96.8 % | **95.5 %** | 86.5 % | 0.76 |

Read this table carefully: the regexes were written against these same three templates, so ~100 %
on the clean set says the baseline fits the generator, **not** that regexes solve invoices. The noisy
set is the more informative one: when OCR garbles a label (`Invoice No` → junk) the rules return
nothing, and 13.5 % of invoices end up with at least one wrong field. Two other failure modes showed up
on the first real run and are now covered by regression tests: a line item called "Tax advisory
session" being read as the VAT line, and `:` OCR'd as `;` in the label.

### Anomaly detection (all 416 invoices)

| Input | Detector | Positives | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ground-truth fields | duplicates | 16 | 1.00 | 1.00 | 1.00 |
| | odd amounts | 16 | 0.81 | 0.81 | 0.81 |
| | misclassified | 20 | 1.00 | 1.00 | 1.00 |
| | any | 52 | 0.94 | 0.94 | 0.94 |
| OCR + rules (end to end) | duplicates | 16 | 1.00 | 1.00 | 1.00 |
| | odd amounts | 16 | 0.81 | 0.81 | 0.81 |
| | misclassified | 20 | 1.00 | 1.00 | 1.00 |
| | any | 52 | 0.94 | 0.94 | 0.94 |

The two blocks are identical because the rules extractor is near-perfect on the clean set; on the
noisy set they would diverge. The odd-amount misses are the three outliers whose multiplier was
close to the low end (z between 2.6 and 2.9, below the threshold); the three false positives are
normal but unusually *small* invoices (z between -3.3 and -4.3), the low tail of a lognormal amount distribution. Lowering the threshold trades one for
the other, so the right setting depends on what a missed anomaly costs versus a wasted review.

Vendor names alone almost determine the category here, which is why the category check is perfect.
Real ledgers are far messier, so treat that number as an upper bound.

## Prompted vs fine-tuned: how to run it (not yet run)

This sandbox had no API key, no GPU and no access to Hugging Face, so these rows are **empty on
purpose** rather than estimated. The code paths are implemented and unit-tested with stub models
(JSON parsing, token/latency bookkeeping, error handling, loss masking, CLI round trip), but no real
model has been run through them.

```bash
# 1. prompted Claude (any model id you have access to)
export ANTHROPIC_API_KEY=...
python -m invoiceai.cli extract --backend claude --model claude-haiku-5-5 --out results/preds_claude.jsonl
python -m invoiceai.cli evaluate --preds results/preds_claude.jsonl --name claude-haiku-prompted \
    --price-in <USD per 1M input tokens> --price-out <USD per 1M output tokens>

# 2. small open model, prompted, then LoRA fine-tuned (pip install -r requirements-llm.txt, GPU recommended)
python -m invoiceai.cli extract --backend local --prompt full --out results/preds_qwen_base.jsonl
python -m invoiceai.cli evaluate --preds results/preds_qwen_base.jsonl --name qwen0.5b-prompted --gpu-hour-cost <USD/h>
python -m invoiceai.finetune_lora --data data --out models/lora-invoice
python -m invoiceai.cli extract --backend local --adapter models/lora-invoice --out results/preds_qwen_lora.jsonl
python -m invoiceai.cli evaluate --preds results/preds_qwen_lora.jsonl --name qwen0.5b-lora --gpu-hour-cost <USD/h>

python -m invoiceai.cli compare        # writes results/comparison.md
```

Prices are arguments on purpose: look up current rates rather than trusting numbers frozen into a
README. No temperature is sent to the API (model default); pass `temperature=` to `ClaudeExtractor`
if you want it pinned for reproducibility.

Things worth checking when you run it:

- **Same test split, same OCR text, same metric** for every row. The fine-tune sees only the train split.
- **Prompt length is part of the cost story.** The full instruction prompt adds ~800 characters
  (roughly 200 tokens) per document on top of ~360 characters of OCR text; the fine-tuned model uses a
  38-character prompt because the task is baked into the weights. That is where fine-tuning saves
  money, if it saves any.
- **Evaluate on the noisy set too.** A fine-tune on clean templates may not hold up there, and that
  is exactly the finding to write down:
  ```bash
  python -m invoiceai.cli generate --data data_hard --n 150 --seed 21 --noise 2.2
  python -m invoiceai.cli ocr --data data_hard
  python -m invoiceai.cli extract --data data_hard --split all --backend local --adapter models/lora-invoice --out results/preds_lora_hard.jsonl
  python -m invoiceai.cli evaluate --data data_hard --split all --preds results/preds_lora_hard.jsonl --name qwen0.5b-lora-hardnoise
  ```
- **Local latency is hardware-dependent** and `--gpu-hour-cost` only counts generation time, not
  loading or idle time. Say which GPU you used.
- Run the fine-tune with a couple of seeds before claiming a difference smaller than a few points;
  184 test documents is small.

## Limitations

- **Synthetic, templated data.** Three layouts, one customer, a fixed vocabulary. Results here show the
  pipeline works, not how it would perform on a real supplier archive. Next step: run `evaluate` on a
  public receipt/invoice set (SROIE, CORD) or a few hundred of your own, anonymised.
- **Text-only extraction.** The LLM sees OCR text, not the image, so table structure is lost and OCR
  failures are unrecoverable. A vision-capable model or a layout-aware model is the obvious comparison.
- **Easy anomalies.** Outliers are 3.5–12× and duplicates are copies. Real duplicates drift (different
  dates, partial amounts, credit notes) and real outliers are subtle.
- **Fixed currencies and no FX.** Amounts are compared within a vendor, which works because each vendor
  bills in one currency here.
- **A flag is a prompt for a human, not a verdict.** Nothing here should auto-reject an invoice.

## Ideas for extending it

- Confidence scores and an "abstain" path for extraction; route low-confidence documents to review.
- Cross-check extracted totals against line items (`Σ lines + VAT = total`) as another anomaly signal.
- Calibrate the amount threshold per vendor from history instead of one global `z`.
- A small review UI (FastAPI + a table) that shows the image next to the extracted fields and flags.
- Track per-field accuracy over time as vendors change layouts.
