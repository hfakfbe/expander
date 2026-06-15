# v07 Phase 5 WikiText Smoke and Real Run Report

Status: passed.

Run location: remote host `huiwei`, `/home/huiwei/ysx/zigzag_attention`.
Code used for Phase 5: `93fce1f`.
Phase 4 data commit recorded in metadata: `dfba8de`.

## Fixed Inputs

- canonical graph sha256: `53ae37a6584833a1d20d51162a01a03d18bf7eeb9fd0efd2ebf3b8a482427a48`
- tokenizer sha256: `1a720397783d6758ee208b8605d7ff197fad86f01b1c56c4c5019cfda96ca616`
- tokenized train sha256: `fbe867482e8ef567a0990477e142af5ae9f12f66ce67fcacd0b32d65d3dd5142`
- tokenized test sha256: `5d9e64748794b682f0cedb4e259a83b646359787df54b9b5f5d2bb26396bb3fa`
- train blocks/tokens: 112540 / 115241250
- test blocks/tokens: 268 / 275123

## Smoke Run

Command log: `logs/wikitext_v07_smoke_n1024_q32_B32_d8_20260614T144847Z.log`.
Output: `outputs/wikitext_v07_smoke_n1024_q32_B32_d8/`.
GPU: `CUDA_VISIBLE_DEVICES=3`.

| method | gpu | steps | lr_scheduler | test_loss | test_ppl | train_tok/s | test_tok/s | peak_alloc_gb | graph_sha | random_k_err_max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| dense | 3 | 2 | constant | 10.446125 | 34410.775 | 4096.9 | 164313.3 | 3.415 | True | 0 |
| local | 3 | 2 | constant | 10.453192 | 34654.825 | 11890.4 | 275025.5 | 3.415 | True | 0 |
| zigzag_certified | 3 | 2 | constant | 10.447923 | 34472.707 | 7641.0 | 31994.5 | 4.280 | True | 0 |
| random_regular | 3 | 2 | constant | 10.450523 | 34562.443 | 8162.6 | 31721.2 | 4.280 | True | 0 |

Smoke checks: all methods status ok; no NaN/non-finite losses; graph/tokenizer/tokenized sha values matched expected; `training_curves.png` exists for every method and metrics include `seconds_since_prev_log`.

## Real Run

Output: `outputs/wikitext_v07_main_n1024_q32_B32_d8/`.
The real run used method-only parallel jobs and was merged into the canonical output directory. Parallel strategy is preserved in `summary.parallel_strategy`.

- first wave: dense on GPU3, local on GPU2, zigzag_certified on GPU1, zigzag_certified_cosine on GPU0
- second wave: random_regular on GPU3, zigzag_boolean on GPU2
- logs: `logs/wikitext_v07_main_n1024_q32_B32_d8_parallel_*_20260614T145205Z.log`
- accounting sum of per-method wall time: 12754.604 sec

| method | gpu | steps | lr_scheduler | test_loss | test_ppl | train_tok/s | test_tok/s | peak_alloc_gb | graph_sha | random_k_err_max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| dense | 3 | 3516 | constant | 5.623853 | 276.955 | 119449.8 | 313474.2 | 13.296 | True | 0 |
| local | 2 | 3516 | constant | 5.579647 | 264.978 | 231434.7 | 596638.2 | 8.929 | True | 0 |
| zigzag_certified | 1 | 3516 | constant | 5.659867 | 287.110 | 41062.8 | 190008.2 | 16.755 | True | 0 |
| zigzag_certified_cosine | 0 | 3516 | cosine | 6.001769 | 404.143 | 40867.2 | 187533.5 | 16.755 | True | 0 |
| random_regular | 3 | 3516 | constant | 5.642595 | 282.194 | 40614.3 | 182627.9 | 16.755 | True | 0 |
| zigzag_boolean | 2 | 3516 | constant | 5.638361 | 281.002 | 40808.5 | 185844.9 | 16.755 | True | 0 |

Real checks: all six methods completed one epoch with `train_steps=3516`; full test split was evaluated with `eval_batches=17`; test loss and perplexity are recorded; graph artifact was copied into every run and sha matched canonical; tokenizer/train/test sha matched Phase 4; random_regular K alignment error max was 0; all training curves exist and include timing/tokens/sec/LR/memory fields; no checkpoint/tensor files were synchronized.

## Notes

`zigzag_certified_cosine` uses the same graph structure as `zigzag_certified` and differs by cosine LR scheduling only. Its final LR reached `3e-05`, matching `min_lr_ratio=0.1` for base LR `0.0003`.
