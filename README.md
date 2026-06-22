# Expander probes

This repository contains the zigzag/expander sparse-attention probe code and the corrected non-causal Copy experiments.

The current Copy pipeline is the corrected copy_corrected_v01 contract from ref/copy_experiment_correction_spec_v01.md:

- input length is fixed at 2048;
- source length is 1024;
- marker/readout length is 1024;
- the model input is source[0:1024] + marker[1024:2048];
- the target is the original source sequence;
- loss is computed only at positions 1024..2047;
- attention is directed non-causal, with causal = false;
- Copy uses non-learnable RoPE on Q/K and no learned absolute position table.

## Environment

Remote experiments use the huiwei SSH alias and the project directory:

~~~bash
ssh huiwei
cd /home/huiwei/ysx/zigzag_attention
source /home/huiwei/miniconda3/etc/profile.d/conda.sh
conda activate ysx_base
~~~

On huiwei, prefer setting one GPU explicitly:

~~~bash
export CUDA_VISIBLE_DEVICES=0
~~~

## Data

Corrected Copy reads from datasets/copy/:

~~~text
datasets/copy/train.jsonl
datasets/copy/test.jsonl
datasets/copy/dataset_card.json
datasets/copy/checksums.sha256
~~~

There is no validation split for corrected Copy. The old validation content is used as the new test split, and the old OOD test is discarded and recorded by hash.

If datasets/copy/train.jsonl or datasets/copy/test.jsonl is missing, materialize them first:

~~~bash
python scripts/materialize_copy_corrected.py \
  --source-dir /Users/sxye/Documents/expander_bench/data/probes/copy/s4_copying_length_extrapolation/copy_s4_l0_m1024_a64_full_v2 \
  --output-dir datasets/copy
~~~

On huiwei, use the corresponding source path if the local macOS path is unavailable.

## Prepare a corrected Copy config

The standard corrected Copy graph satisfies:

~~~text
T = 2048
q * B = 2048
0 < d < B
~~~

For the common q=32, B=64 setting:

~~~bash
python scripts/prepare_copy_corrected.py \
  --skip-branch-check \
  --data-dir datasets/copy \
  --output-root outputs/copy_corrected_q32_B64_d32_l8_log5 \
  --config configs/copy_corrected_q32_B64_d32_l8_log5.json \
  --manifest configs/copy_corrected_q32_B64_d32_l8_log5_task_parameters.json \
  --layers 8 \
  --log-every 5 \
  --q 32 \
  --block-size 64 \
  --degree 32 \
  --graph-seed 0 \
  --methods zigzag_certified random_regular \
  --trial-id q32_B64_d32_l8_log5 \
  --version copy_corrected_q32_B64_d32_l8_log5
~~~

Parameter options accepted by prepare_copy_corrected.py:

- --data-dir: corrected Copy data directory, default datasets/copy.
- --output-root: artifact root containing encoder and graph artifacts.
- --config: JSON config path to write.
- --manifest: task parameter manifest path to write.
- --layers: Transformer layer count.
- --log-every: training diagnostic log interval.
- --block-size: local block size B.
- --q: number of blocks; must satisfy q * B = 2048.
- --degree: zigzag/random remote degree d; must satisfy 0 < d < B.
- --graph-seed: graph artifact seed.
- --methods: methods written into the config.
- --trial-id: output run directory name under output_root/runs.
- --version: experiment version string.
- --branch-name: provenance label.
- --skip-branch-check: allow running from main/deployed directories instead of the old corrected-copy worktree branch.

## Methods

The Copy runner accepts these method names:

- dense: full non-causal attention.
- local: block-local non-causal attention with block size B.
- zigzag_certified: local block + zigzag remote mask using multiplicity/log-m weighting, named unique_log_m in older result fields.
- zigzag_boolean: local block + zigzag remote mask with boolean edges, ignoring multiplicity weights.
- random_regular: local block + random remote keys. By default it aligns each query's non-causal unique K to the zigzag mask.
- zigzag_cycle: older cyclic zigzag baseline.
- Aliases in graph utilities: random maps to random_regular, and zigzag maps to zigzag_cycle.

Use --method METHOD to run only one method; omit it to run the config's methods list.

## Run corrected Copy

Runner options:

- --config: config JSON.
- --mode: one of gate-overfit, train, final-eval.
- --method: optional method override.
- --seed: optional seed override.
- --device: auto, cpu, cuda, or mps.
- --checkpoint: checkpoint path for final-eval; if omitted, the runner reads the latest train checkpoint from the train run directory.

Gate-overfit sanity check:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d32_l8_log5.json \
  --mode gate-overfit \
  --method dense \
  --seed 0 \
  --device cuda
~~~

Train one method:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d32_l8_log5.json \
  --mode train \
  --method zigzag_certified \
  --seed 0 \
  --device cuda
~~~

Evaluate the latest train checkpoint:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d32_l8_log5.json \
  --mode final-eval \
  --method zigzag_certified \
  --seed 0 \
  --device cuda
~~~

If evaluating a gate-overfit checkpoint or a manually chosen checkpoint, pass it explicitly:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d32_l8_log5.json \
  --mode final-eval \
  --method dense \
  --seed 0 \
  --device cuda \
  --checkpoint outputs/copy_corrected_q32_B64_d32_l8_log5/runs/q32_B64_d32_l8_log5/gate_overfit/dense/seed0/checkpoints/gate-overfit_final_step1000.pt
~~~

## Existing Copy configs

Main q32/B64/d32 8-layer config:

~~~bash
python scripts/run_copy_corrected.py --config configs/copy_corrected_q32_B64_d32_l8_log5.json --mode train --method zigzag_certified --seed 0 --device cuda
python scripts/run_copy_corrected.py --config configs/copy_corrected_q32_B64_d32_l8_log5.json --mode train --method random_regular --seed 0 --device cuda
~~~

Random density configs, shared mask across layers:

~~~bash
for density in 10 50 80 90; do
  python scripts/run_copy_corrected.py \
    --config configs/copy_corrected_q32_B64_d32_l8_log5_random_density${density}.json \
    --mode train --method random_regular --seed 0 --device cuda
done
~~~

Random density configs, independent mask per layer:

~~~bash
for density in 10 30 50 70 90; do
  python scripts/run_copy_corrected.py \
    --config configs/copy_corrected_q32_B64_d32_l8_log5_random_layerwise_density${density}.json \
    --mode train --method random_regular --seed 0 --device cuda
done
~~~

Random 12-layer 10% density:

~~~bash
python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d32_l12_log5_random_density10.json \
  --mode train --method random_regular --seed 0 --device cuda
~~~

Zigzag layerwise-random graph, about 10% density, layers 8 and 12:

~~~bash
python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l8_log5_zigzag_layerwise_density10_multiplicity_logm.json \
  --mode train --method zigzag_certified --seed 0 --device cuda

python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l8_log5_zigzag_layerwise_density10_boolean_unique.json \
  --mode train --method zigzag_boolean --seed 0 --device cuda

python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l12_log5_zigzag_layerwise_density10_multiplicity_logm.json \
  --mode train --method zigzag_certified --seed 0 --device cuda

python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l12_log5_zigzag_layerwise_density10_boolean_unique.json \
  --mode train --method zigzag_boolean --seed 0 --device cuda
~~~

Zigzag shared graph across layers, about 10% density, layers 8 and 12:

~~~bash
python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l8_log5_zigzag_shared_density10_multiplicity_logm.json \
  --mode train --method zigzag_certified --seed 0 --device cuda

python scripts/run_copy_corrected.py \
  --config configs/copy_corrected_q32_B64_d13_l12_log5_zigzag_shared_density10_multiplicity_logm.json \
  --mode train --method zigzag_certified --seed 0 --device cuda
~~~

## Result locations

Training/eval outputs are written under the config's output_root, for example:

~~~text
outputs/copy_corrected_q32_B64_d32_l8_log5/runs/<trial_id>/<method>/seed0/
~~~

Checkpoints are under checkpoints/ and are git-ignored. The JSON/CSV summaries, metrics, config snapshots, graph artifacts, and reports are kept as normal experiment artifacts.

The v08/corrected-run archive index is in ref/archive_v08_complete_20260622/. Large local tarball payloads under ref/archive_v08_complete_*/snapshots/*.tar.gz are intentionally ignored by git; the archive README, file lists, and checksums are tracked.
