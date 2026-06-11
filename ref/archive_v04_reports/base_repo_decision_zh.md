# 基础仓库决策

日期：2026-06-10

## 决策

使用 `guy-dar/lra-benchmarks` 作为第一条实现路径的基础仓库。

远程检出位置：

```text
/home/huiwei/ysx/zigzag_attention/code/lra-benchmarks
```

基础提交：

```text
afcf5c1834ca0a0ad42ddd0684141bd1ce30f2b7
```

## 选择该基础仓库的原因

`guy-dar/lra-benchmarks` 最符合手册的 phase-1 标准：

- 基于 PyTorch，因此更容易为自定义 attention 做修改。
- 使用 HuggingFace `BertForSequenceClassification`，attention 替换路径清晰。
- task/config/dataset 文件较紧凑。
- 可以在 `ysx_base` 中的单张 A100 上运行。
- 100-step GPU smoke test 已成功完成。

## 被拒绝 / 暂缓的选项

| 仓库 / 资源 | 决策 | 原因 |
| --- | --- | --- |
| `google-research/long-range-arena` | 暂作为参考 | 官方仓库，但使用 JAX/Flax；对本 PyTorch 实验而言修改成本更高。 |
| Hugging Face BigBird | 暂作为 baseline/reference | 有用的 baseline，但本身不是完整的实验基础。 |
| `google-research/bigbird` | 仅作参考 | sparse attention 实现参考，不作为第一训练基础。 |
| `facebookresearch/xformers` | 后续优化 | 在正确性和质量建立之后会有用。 |
| `hamed1375/Exphormer` | 仅作参考 | expander 思路相关，但 graph-task 语境不是主要 sequence 基础。 |
| OpenAI sparse_attention | 历史参考 | 实现思路有用，但不是当前 PyTorch LRA 基础。 |
| Longformer | baseline/reference | 有用的 local/global baseline，不选为主基础。 |

## Attention 修改位置

第一个 attention 集成点位于底层的 HuggingFace BERT attention：

```text
run_model.py
  get_model(...)
    BertConfig(...)
    BertForSequenceClassification(...)
```

对于 phase 3/4 实现，可能的修改路径是：

1. 子类化或包装 `BertSelfAttention` / `BertSdpaSelfAttention`，并在模型构造后替换 encoder layer attention。
2. 如果 HuggingFace 内部结构让 mask 替换过于脆弱，则构建一个小型 local Transformer encoder，并保持相同的 dataset/config 接口。
3. 保留 `lra_datasets.py`、tokenizers、task config 约定、logging 和 evaluation 形状，只替换 model module。

仍应遵循手册要求的实现顺序：

```text
dense mask debug -> neighbor list attention -> local + cross split -> cached graph -> layer-wise graph variant
```

## Smoke Test 命令和结果

命令：

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

结果摘要：

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

该 smoke test 验证了：

- 程序启动。
- 通过基础 `ListOpsDataset` 路径加载数据集，使用生成的 tiny ListOps-format TSV 文件。
- 通过基础 `get_model` 构造模型。
- 前向传播。
- 反向传播。
- Optimizer step。
- 在 A100 上执行 CUDA。
- 输出 log 和 JSON metric。

## Phase 2 前的注意事项

- 完整 LRA ListOps 数据尚未在本地可用，因为服务器无法访问文档中的 Google Storage URL。
- `fetch_data.py` 需要一个小 bug fix，或应通过可靠的下载/准备脚本绕过。
- `ysx_base` 中缺少 `ml_collections`；应使用私有环境或 project-local shim，而不是修改公共包。
- 每次运行前重新检查 GPU occupancy；初次 idle-GPU 检查后又出现了其他进程。
