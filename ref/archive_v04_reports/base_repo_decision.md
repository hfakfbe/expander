# Base Repository Decision

Date: 2026-06-10

## Decision

Use `guy-dar/lra-benchmarks` as the base repository for the first implementation path.

Remote checkout:

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
```

Base commit:

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

## Why This Base

`guy-dar/lra-benchmarks` is the best fit for the manual's phase-1 criteria:

- PyTorch-based and therefore easier to edit for custom attention.
- Uses HuggingFace `BertForSequenceClassification`, giving a clear attention replacement path.
- Has compact task/config/dataset files.
- Can run on a single A100 in `ysx_base`.
- A 100-step GPU smoke test completed successfully.

## Rejected / Deferred Options

| Repository / Resource | Decision | Reason |
| --- | --- | --- |
| `google-research/long-range-arena` | Defer to reference | Official but JAX/Flax, higher modification cost for this PyTorch experiment. |
| Hugging Face BigBird | Defer to baseline/reference | Useful baseline but not a complete experiment base by itself. |
| `google-research/bigbird` | Reference only | Sparse attention implementation reference, not first training base. |
| `facebookresearch/xformers` | Later optimization | Useful after correctness and quality are established. |
| `hamed1375/Exphormer` | Reference only | Expander idea is relevant, but graph-task context is not the main sequence base. |
| OpenAI sparse_attention | Historical reference | Useful implementation ideas, not a current PyTorch LRA base. |
| Longformer | Baseline/reference | Useful local/global baseline, not selected as the main base. |

## Attention Modification Location

The first attention integration point is HuggingFace BERT attention underneath:

```text
run_model.py
  get_model(...)
    BertConfig(...)
    BertForSequenceClassification(...)
```

For phase 3/4 implementation, likely modification paths are:

1. Subclass or wrap `BertSelfAttention` / `BertSdpaSelfAttention` and replace encoder layer attention after model construction.
2. Build a small local Transformer encoder with the same dataset/config interface if HuggingFace internals make mask replacement too brittle.
3. Keep `lra_datasets.py`, tokenizers, task config conventions, logging, and evaluation shape while replacing only the model module.

The manual's required implementation order should still be followed:

```text
dense mask debug -> neighbor list attention -> local + cross split -> cached graph -> layer-wise graph variant
```

## Smoke Test Command And Result

Command:

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

Result summary:

```text
status: ok
steps: 100
device: cuda
gpu_name: NVIDIA A100-SXM4-80GB
torch_version: 2.10.0+cu128
final_loss: 1.0127418041229248
mean_loss: 1.465013245344162
tokens_per_sec: 18974.068654582854
elapsed_sec: 2.698419665917754
peak_allocated_gb: 0.0232696533203125
peak_reserved_gb: 0.02734375
```

The smoke test validates:

- Program startup.
- Dataset loading through the base `ListOpsDataset` path, using generated tiny ListOps-format TSV files.
- Model construction through base `get_model`.
- Forward pass.
- Backward pass.
- Optimizer step.
- CUDA execution on an A100.
- Log and JSON metric output.

## Caveats Before Phase 2

- Full LRA ListOps data is not yet available locally because the documented Google Storage URL failed from the server.
- `fetch_data.py` needs a small bug fix or should be bypassed with a reliable download/preparation script.
- `ml_collections` is absent from `ysx_base`; use a private environment or project-local shim rather than changing public packages.
- Re-check GPU occupancy before every run; other processes appeared after the initial idle-GPU check.
