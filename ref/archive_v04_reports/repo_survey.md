# Repository Survey

Date: 2026-06-10

Execution context:

- Local workspace: `/Users/sxye/Documents/expander`
- Remote server: `huiwei`
- Remote project directory: `/home/huiwei/ysx/zigzag_attention`
- Conda executable: `/home/huiwei/miniconda3/bin/conda`
- Remote environment: `ysx_base`

## Candidate Summary

| Priority | Repository / Resource | Status | Notes |
| --- | --- | --- | --- |
| A | `guy-dar/lra-benchmarks` | Selected as base | PyTorch/HuggingFace style, small code surface, clear `run_model.py`, dataset classes, and BERT attention path. Remote clone and GPU smoke test succeeded. |
| B | `google-research/long-range-arena` | Reference only | Official LRA repository, but JAX/Flax oriented and less suitable for fast attention-module edits in this project. HEAD was reachable by `git ls-remote`. |
| C | Hugging Face BigBird | Future baseline/reference | Useful for BigBird-style sparse baseline and block sparse design, but not the first base because task/training framework still needed. |
| D | `google-research/bigbird` | Reference only | Useful for sparse attention structure, not selected as primary sequence experiment base. |
| E | `facebookresearch/xformers` | Future performance reference | Useful after correctness and MVP quality experiments, especially for optimized attention backends. |
| F | `hamed1375/Exphormer` | Reference only | Relevant expander design ideas, but graph Transformer context does not match the sequence LRA base as directly. |
| G | OpenAI sparse_attention | Reference only | Historical fixed/strided sparse patterns, not selected as base. |
| H | Longformer | Reference/baseline | Useful local/global baseline idea, not selected as primary base. |

## Checks Performed

### Remote Server

`ssh huiwei` succeeded. Initial GPU query showed four NVIDIA A100-SXM4-80GB devices visible. Later, other user processes appeared on the GPUs, so future runs must re-check `nvidia-smi` immediately before launching training.

`ysx_base` resolved through `/home/huiwei/miniconda3/bin/conda` rather than the default shell PATH. Key runtime facts:

- Python: 3.10.0
- PyTorch: 2.10.0+cu128
- CUDA visible from PyTorch: yes
- GPU count: 4
- Device name: NVIDIA A100-SXM4-80GB

### `guy-dar/lra-benchmarks`

Remote clone:

```bash
cd /home/huiwei/ysx/zigzag_attention/code
git clone --depth 1 https://github.com/guy-dar/lra-benchmarks.git
cd lra-benchmarks
git rev-parse HEAD
```

Commit:

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

The repository code is small and readable:

- `run_model.py`: model construction, training loop, task registry.
- `lra_config.py`: task/model configs and tokenizers.
- `lra_datasets.py`: ListOps, CIFAR-10, IMDB dataset loaders.
- `train_utils.py`: learning-rate schedules.

The base repository remained clean after smoke testing:

```bash
cd /home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
git status --short
```

No source changes were made inside the base checkout.

## Issues Found

1. `ysx_base` does not include `ml_collections`.

   To avoid modifying the public environment, the smoke test used a project-local compatibility shim in:

   ```text
   /home/huiwei/ysx/zigzag_attention/code/smoke_scripts/smoke_support/ml_collections
   ```

2. `fetch_data.py` has a bug in the ListOps branch.

   It indexes `task["lra_release"]["url"]` after assigning `task = args.task`, so `task` is a string and the script raises:

   ```text
   TypeError: string indices must be integers
   ```

3. The documented LRA data URL was not reliably usable from the server.

   Attempting to download `https://storage.googleapis.com/long-range-arena/lra_release.gz` returned `403 Forbidden` or stalled. Hugging Face dataset probing also timed out on the server. Therefore, the phase-1 smoke test used a generated tiny ListOps-format TSV to validate repository trainability without claiming that full LRA data is prepared.

4. Local machine GitHub clone failed intermittently with TLS errors.

   Remote clone on `huiwei` succeeded and is the authoritative base checkout for this phase.

## Smoke Test Result

Smoke test command:

```bash
cd /home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../smoke_scripts/smoke_support:/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python ../smoke_scripts/base_smoke_listops.py \
  --repo-dir . \
  --steps 100 \
  --rows 256 \
  --batch-size 4 \
  --max-length 128 \
  --hidden-size 64 \
  --layers 2 \
  --heads 4 \
  --output-json ../../outputs/smoke/base_listops_smoke.json \
2>&1 | tee ../../logs/base_listops_smoke.txt
```

Result:

- Status: OK
- Steps: 100
- Device: CUDA
- GPU: NVIDIA A100-SXM4-80GB
- Final loss: 1.0127418041229248
- Mean loss: 1.465013245344162
- Tokens/sec: 18974.068654582854
- Peak allocated memory: 0.0232696533203125 GB
- Peak reserved memory: 0.02734375 GB

Artifacts:

- `smoke_test_log.txt`
- `outputs/base_listops_smoke.json`
- `env_snapshot.yaml`
- `envs/requirements_snapshot.txt`

## Phase-1 Decision

`guy-dar/lra-benchmarks` passes the phase-1 smoke-test requirement as a reusable base repository, with caveats:

- Full LRA data is not yet prepared because the upstream data path is currently unreliable.
- The base needs either `ml_collections` installed in a private environment or the project-local shim retained for smoke/debug scripts.
- `fetch_data.py` should be patched or bypassed in phase 2.

The next phase should prepare stable datasets and task configs, starting with synthetic Associative Recall / Delayed Copy and then resolving ListOps through a reliable mirror or generated-compatible local preprocessing.
