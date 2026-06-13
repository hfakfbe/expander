# WikiText2 v06 Pipeline Report

Date: 2026-06-13

This report records the WikiText2 branch required by
`ref/zigzag_experiment_execution_manual_v06.md`.

## Data Readiness

- Dataset source: `Salesforce/wikitext`
- Variant: `wikitext-2-raw-v1`
- Revision/hash: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Download path: parquet files from the Hugging Face dataset repository.
- Output dir: `datasets/wikitext2_raw_v1`
- Tokenizer: project-local byte-level UTF-8 tokenizer, vocab size `258`, PAD `0`, EOS `1`.

| Split | Rows | Non-empty rows | Empty-line rate |
|---|---:|---:|---:|
| train | 36718 | 23767 | 0.35271529 |
| validation | 3760 | 2461 | 0.34547872 |
| test | 4358 | 2891 | 0.33662230 |

Data artifacts:

- `datasets/wikitext2_raw_v1/dataset_info.json`
- `datasets/wikitext2_raw_v1/data_readiness.json`
- `datasets/wikitext2_raw_v1/tokenized_smoke.json`
- `datasets/wikitext2_raw_v1/train.jsonl`
- `datasets/wikitext2_raw_v1/validation.jsonl`
- `datasets/wikitext2_raw_v1/test.jsonl`

## Smoke

Smoke command:

```bash
python scripts/wikitext2_smoke.py --config configs/wikitext2_v06_smoke.json --output-dir outputs/wikitext2_v06_smoke --device cpu
```

Smoke status: `ok`.

| Method | train loss finite | validation loss finite | backward |
|---|---:|---:|---:|
| dense | true | true | true |
| local | true | true | false |
| random_regular | true | true | false |
| zigzag_certified | true | true | false |

The selected copy graph artifact has `T=1040`; WikiText2 uses effective LM
`sequence_length=512` and pads the input to `T=1040` so the selected artifact is
used without regenerating a separate graph.

## Pipeline Eval

Eval command:

```bash
python scripts/wikitext2_eval.py --config configs/wikitext2_v06_eval.json --output-dir outputs/wikitext2_v06_eval --device cpu
```

Eval status: `ok`; `pipeline_only=true`. No training quality conclusion is made
because `steps=0`.

| Method | validation loss | validation ppl | test loss | test ppl |
|---|---:|---:|---:|---:|
| dense | 5.65089798 | 284.54686846 | 5.67657948 | 291.94910099 |
| local | 5.65358591 | 285.31273966 | 5.68056917 | 293.11621597 |
| random_regular | 5.65317631 | 285.19589861 | 5.67886734 | 292.61780569 |
| zigzag_certified | 5.64977407 | 284.22724440 | 5.67691088 | 292.04586949 |

Eval artifacts:

- `outputs/wikitext2_v06_eval/summary.json`
- `outputs/wikitext2_v06_eval/results.csv`
- `outputs/wikitext2_v06_eval/results.jsonl`
- `outputs/wikitext2_v06_eval/metrics.jsonl`
- `outputs/wikitext2_v06_eval/data_readiness.json`
- `outputs/wikitext2_v06_eval/command.sh`
- `outputs/wikitext2_v06_eval/config_snapshot.json`

## Limits

- This is a real-text pipeline check, not an official benchmark.
- The eval is pipeline-only (`steps=0`), so validation/test losses are sanity values for an untrained small model.
- No general sparse-attention advantage or language-model quality claim is made.
