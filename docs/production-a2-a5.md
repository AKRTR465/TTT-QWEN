# A2 → A5 四卡生产训练

本页描述当前生产入口。P15–P18 文档保留的是历史 synthetic/tiny 工程门禁，不再代表正式训练配置。
正式路径只有 A2 和 A5；A3/A4 仅保留为消融代码，生产 YAML 无法到达。

## 已实现的边界

- A2 全量解冻 Qwen ViT、Main Merger、DeepStack merger、36 层 Decoder，并训练状态模块和
  `W0`；Predictor 冻结，不运行 Inner SGD，目标严格为 `L_state + L_answer`。
- A5 对 Support 数不设上限，按处理过的 Support 每 `K=8` 步截断。段内
  `create_graph=True`，截断点使用
  `stopgrad(W_t) + W0 - stopgrad(W0)`，保留数值状态并重新建立到 `W0` 的梯度路径。
- 非末段立即 backward `0.1/T * sum(L_TTT)`；最后一段和所有 Query 一起 backward。
  一个 episode 只由外层 Trainer 裁剪和执行一次 AdamW step。
- Inner SGD 的唯一参数是 transient `w_t_1/w_t_2`，momentum 固定为 0。Qwen、状态模块、
  `W0` 和 Predictor 只能进入 Outer AdamW。
- hard Bank/FSM commit 与 soft observation forward 分离；activation checkpoint 重算只能经过
  soft 路径。
- 每一步只物化一个当前 chunk；视觉帧数与 Token 数可动态变化，历史 chunk 不进入 Qwen
  上下文，只通过 `W_t`、Bank/FSM 和受限时序状态延续。
- manifest、采样器、optimizer 参数组、A2→A5 权重初始化和 checkpoint 边界由 TTT-QWEN
  中央控制，不修改相邻的 LLaMA-Factory 工作树。

## 数据准备

H200 已有的转换集为“每个 Query 一份因果视频”。准备脚本会校验 4576 个 Query 的映射，
A2 使用对应 Query clip，A5 使用组内最后一个 Query clip，并从每个原始视频的最后 clip 自动
读取容器时长：

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen
export PYTHONPATH="$PWD/src"

$PWD/.venv-h200/bin/python scripts/prepare_svcbench_episodes.py \
  --annotation /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part/raw/data__vcbench_data.jsonl \
  --converted-dataset /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part/svcbench_qwen3vl_sft.json \
  --video-root /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part \
  --dataset-name svcbench-part \
  --dataset-revision h200-20260710 \
  --output-root runs
```

脚本创建独立的 `MMDD_HHMMSS_prepare_svcbench_k8/`，写出 `dataset_manifest.json`、
`failed.jsonl`、`succeeded.jsonl`、`run_config.json`、`run_summary.json` 和
`experiment.log`。manifest 固定 `fold0/seed=42`，按原始视频切分，使用 64 秒 greedy
Query 分组、细粒度近历史加几何扩宽远历史、每区间最多 16 帧，并为四卡按
`tbptt_segment_count` 生成零权重 padding。

监督在物理上分成 `runtime`、`answer`、`weak` 三个 sidecar。中央 loader 会拒绝 runtime
中出现 `answer/count/occurrence/counting_type/counting_subtype` 等字段；loss builder 只能在
forward 完成后读取后两个 sidecar。

## 内置 production runtime

生产 YAML 固定使用 `ttt_svcbench_qwen.production_runtime:build_runtime`，无需用户提供外部
factory。中央 bridge 会覆盖 runtime 的 dataset 字段，强制使用 manifest 的 train/validation
视图。

运行时边界为：

- 返回模型注册加载得到的同一个 `backbone.model`，并注册状态模块、`W0` 和 Predictor；
- A2 返回 `stage_a_loss_step`，且 Predictor 全冻结；A5 返回 `MetaTTTEpisodeRunner` 与
  `episode_adapter`，且 Predictor 可训练；
- collator 接收 `A2QueryRecord` 或 `A5EpisodeRecord`，Support 先保持轻量时间区间，执行到该步
  才解码并处理当前 chunk；
- Query/weak/answer sidecar 的 join 发生在 forward 后；
- A5 padding episode 必须完整执行同数目的 backward collective，但返回 `loss_weight=0`；
- 不把 transient `W_t`、Bank、FSM、时序/视觉 cache 注册成 parameter 或 buffer。

入口会在训练前审计以上关键参数边界，不会退回普通 SFT。

## 四卡运行

在四卡 worker 内直接运行；首次执行会在项目 `.venv-h200-py312` 建立 Python 3.12 环境，复用
worker 已有的 CUDA 12.8 PyTorch，并从 PJLAB 镜像安装其余锁定依赖。省略 manifest 时会使用
远端现有 SVCBench 转换集自动生成：

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen
bash scripts/h200/launch_qwen3vl8b_ttt_a2_full4.sh
```

A2 成功后，使用最后 epoch 的完整模型权重初始化 A5；不会继承 A2 optimizer、scheduler 或
Trainer step：

```bash
bash scripts/h200/launch_qwen3vl8b_ttt_a5_k8_full4.sh \
  /absolute/path/a2_run/checkpoints/final-checkpoint \
  /absolute/path/dataset_manifest.json
```

启动脚本要求当前用户为 `niujunbo`、至少 4 张可见 GPU、共享盘至少 200 GiB 空闲；它先做
manifest 严格加载和共享盘 safetensors 往返 smoke，再创建唯一 run 目录并执行四进程训练。
它不会配置 Mac 的本地代理，也不会写入 dirty 的 LLaMA-Factory checkout。

## Checkpoint 与续训

- A2 每 2 epoch 保存一个标准 Trainer checkpoint，A5 每 epoch 保存一个；这些
  `checkpoint-*` 包含模型、optimizer、scheduler、Trainer state 和分布式 RNG，可用于同阶段续训。
- 同阶段续训必须新建 run，并显式设置
  `TTT_RESUME_CHECKPOINT=/old/run/checkpoints/checkpoint-N`。入口校验 checkpoint 的 stage 与
  `run_config.json` 一致。
- A2→A5 是阶段切换，只使用 `A2_CHECKPOINT` 中的模型/module 权重，并创建全新的 A5
  optimizer/scheduler/RNG。
- `final-checkpoint/` 保存最终模型；其 `resume_state/` 还保存 Accelerator 的完整分布式归档状态。
  原生 Trainer 续训应使用同阶段的标准 `checkpoint-*`，不要把 `resume_state/` 当成
  `resume_from_checkpoint`。
- transient `W_t`、Bank、FSM、视觉/时序 cache 从所有 checkpoint 中排除。

同阶段续训示例：

```bash
export TTT_RESUME_CHECKPOINT=/absolute/path/old_run/checkpoints/checkpoint-20
bash scripts/h200/launch_4gpu.sh a5
```

## 验收入口

```bash
python -m pytest -q \
  tests/test_fast_ttt.py \
  tests/test_meta_trainer.py \
  tests/test_episode_data.py \
  tests/test_stage_a_targets.py \
  tests/test_production_factory.py

bash -n scripts/h200/launch_4gpu.sh
```

CPU 测试覆盖 `T=17/K=8`、两次历史截断、数值连续、旧图断开、`W0` 重锚梯度、严格两矩阵
Inner 参数、manifest 防泄漏、四 rank segment parity、完整 checkpoint 边界与同阶段 resume
校验。真实单卡 2B、四卡 8B、显存平台、20-episode resume 和正式训练仍必须在 H200 上执行并
把证据写入各自 run 目录。
