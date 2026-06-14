# v07 Phase 4 WikiText Data and Tokenization Report

Status: passed.

## Command

```bash
cd /home/huiwei/ysx/zigzag_attention
source /home/huiwei/miniconda3/etc/profile.d/conda.sh
conda activate ysx_base
GIT_COMMIT=dfba8de python scripts/prepare_wikitext.py --config configs/wikitext_v07_data_tokenize_n1024.json --output-dir outputs/wikitext_v07_data_tokenize_n1024 2>&1 | tee logs/wikitext_v07_data_tokenize_n1024_20260614T143028Z.log
```

Run date: 2026-06-14 UTC.
Run location: remote host `huiwei`, `/home/huiwei/ysx/zigzag_attention`.
Code used for the run: `dfba8de`. Metadata was post-run corrected from empty `git_commit` to `dfba8de` because that deployed archive had no `.git` directory; commit fallback was subsequently fixed in `aeeb83e`.

## Dataset

- dataset: wikitext-103-raw-v1
- source: Salesforce/wikitext
- config: wikitext-103-raw-v1
- revision/hash: not reported by HF API during run
- local cache/path: datasets/wikitext_v07_raw
- train rows/nonempty: 1801350 / 1165029
- test rows/nonempty: 4358 / 2891

## Tokenizer And Blocks

- tokenizer: byte_level_bpe, vocab 32000, min_frequency 2
- tokenizer train split: train; test split was not used for tokenizer training
- tokenizer sha256: 1a720397783d6758ee208b8605d7ff197fad86f01b1c56c4c5019cfda96ca616
- sequence_length: 1024
- train blocks/tokens: 112540 / 115241250
- test blocks/tokens: 268 / 275123
- tokenized train sha256: fbe867482e8ef567a0990477e142af5ae9f12f66ce67fcacd0b32d65d3dd5142
- tokenized test sha256: 5d9e64748794b682f0cedb4e259a83b646359787df54b9b5f5d2bb26396bb3fa

## Timing

- data download/load: 154.794 sec
- tokenizer train: 28.730 sec
- tokenization/write: 357.816 sec
- total: 577.495 sec

## Git Policy

The raw WikiText files and tokenized train JSONL are large and are not committed. Their remote paths, sizes, and sha256 values are recorded in `outputs/wikitext_v07_data_tokenize_n1024/large_artifacts_manifest.json`. The tokenizer, metadata, tokenized test blocks, configs with fixed sha256 checks, and official log are committed.

## Pass Checks

- train/test split nonempty: yes
- tokenizer trained only on train split: yes
- tokenizer sha256 nonempty: yes
- tokenized train/test sha256 nonempty: yes
- tokenized train/test files exist on remote: yes
- sequence_length = 1024: yes
- train/test block counts > 0: yes
- total_wall_time_sec present: yes

