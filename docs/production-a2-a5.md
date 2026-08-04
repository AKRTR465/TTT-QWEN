# A2 → A5 单节点 4/8 卡生产训练

本页描述当前生产入口。正式路径只有 A2 和 A5；历史阶段 gate、standalone trainer 与
synthetic ablation harness 已从主线删除。

## 已实现的边界

- A2 全量解冻 Qwen ViT、Main Merger、DeepStack merger、36 层 Decoder，并训练状态模块和
  `W0`；Associative 投影冻结，memory 写入不可达，目标严格为 `L_state + L_answer`。
- A5 对 Support 数不设上限，按处理过的 Support 每 `K=8` 步截断。段内 meta 图沿闭式
  delta-rule 线性递推构建（无 `create_graph` 二阶项），截断点对 memory `M` 执行
  detach-保值（`truncate_memory_state`），无 W0 重锚——memory 每视频零初始化，
  没有需要保留的 W0 血统。
- 每个 Support hard commit 后，adapter 从已提交 soft state 派生至多 32 条 slot
  (key, value, eta) 写入对并执行一次并行 delta-rule 写入
  `M ← (1-β)M + Ση(v - Mk)kᵀ`；写入只更新 transient `M`，不执行 Support auxiliary
  backward。一个 episode 只由外层 Trainer 裁剪和执行一次 AdamW step。
- 在线唯一可变状态是 transient `M`（零初始化、Ση ≤ 1 收缩界）。Qwen、状态模块、`W0`、
  `P_C` 与 memory 接口参数只能进入 Outer AdamW；Query Answer/State loss 是唯一 Outer
  objective。
- hard Bank/FSM commit 与 soft observation forward 分离；activation checkpoint 重算只能经过
  soft 路径。
- Support 每一步只物化一个 8/16 帧动态 chunk，处理后不保留历史视觉 Token；Query 单独从
  `[0, query_time]` 以 2 FPS 采样，超过 256 帧时按 LLaMA-Factory uniform-cap 规则降至
  256 帧。256 限制帧数而非视觉 Token 数。
- manifest、采样器、optimizer 参数组、A2→A5 权重初始化和 checkpoint 边界由 TTT-QWEN
  中央控制，不修改相邻的 LLaMA-Factory 工作树。

## 数据准备

H200 已有的转换集为“每个 Query 一份因果视频”。准备脚本用它校验 4576 个 Query 的映射，
实际 A2/A5 runtime 使用原始连续视频：Support 按自适应窗口读取，Query 在原视频上严格裁到
`query_time`。任何超过原视频容器时长的 Query 写入 `failed.jsonl`：

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen
export PYTHONPATH="$PWD/src"

$PWD/.venv-h200/bin/python scripts/prepare_svcbench_episodes.py \
  --annotation /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part/raw/data__vcbench_data.jsonl \
  --converted-dataset /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part/svcbench_qwen3vl_sft.json \
  --video-root /mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/SVCBench/videos \
  --dataset-name svcbench-part \
  --dataset-revision h200-20260710 \
  --output-root runs
```

脚本创建独立的 `MMDD_HHMMSS_prepare_svcbench_k8_w<world_size>/`，写出 `dataset_manifest.json`、
`failed.jsonl`、`succeeded.jsonl`、`run_config.json`、`run_summary.json` 和
`experiment.log`。manifest 固定 `fold0/seed=42`，按原始视频切分，使用 64 秒 greedy
Query 分组、细粒度近历史加几何扩宽远历史、每区间最多 16 帧，并按 `--world-size`（4 或 8）
为相同 `tbptt_segment_count` 生成零权重 padding；8 卡 warmup 不可复用四卡 manifest。

监督在物理上分成 `runtime`、`answer`、`weak` 三个 sidecar。中央 loader 会拒绝 runtime
中出现 `answer/count/occurrence/counting_type/counting_subtype` 等字段；loss builder 只能在
forward 完成后读取后两个 sidecar。

## 内置 production runtime

生产 YAML 固定使用 `ttt_svcbench_qwen.production_runtime:build_runtime`，无需用户提供外部
factory。中央 bridge 会覆盖 runtime 的 dataset 字段，强制使用 manifest 的 train/validation
视图。

运行时边界为：

- 返回模型注册加载得到的同一个 `backbone.model`，并注册状态模块、`W0` 和 Associative 投影；
- A2 返回 `stage_a_loss_step`，且 `P_C` 冻结；A5 返回 `MetaTTTEpisodeRunner` 与
  `episode_adapter`，且 `P_C` 可训练；
- collator 接收 `A2QueryRecord` 或 `A5EpisodeRecord`，Support 先保持轻量时间区间，执行到该步
  才解码并处理当前 chunk；
- Query/weak/answer sidecar 的 join 发生在 forward 后；
- A5 padding episode 必须完整执行同数目的 backward collective，但返回 `loss_weight=0`；
- 不把 transient `M`、Bank、FSM、时序/视觉 cache 注册成 parameter 或 buffer。

入口会在训练前审计以上关键参数边界，不会退回普通 SFT。

## 单节点 4/8 卡运行

在单节点 4/8 卡 worker 内直接运行。full-prefix 入口只使用现有
`.venv-h200-py312-torch28`，不会在线安装依赖。省略 manifest 时会用远端 SVCBench 数据自动
生成：

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen
bash scripts/h200/train_fullprefix256.sh a2
```

A2 成功后，先运行独立 256-step Memory/State Warmup。Warmup 完全冻结 Qwen、W0 和
RMSNorm/P_in/P_out，只训练 P_C、memory 接口和四个 state 组；它只保存非 Qwen handoff
bundle，不继承 A2 optimizer、scheduler 或 Trainer step：

```bash
bash scripts/h200/train_a5_fast_state_warmup.sh \
  /absolute/path/a2_run/checkpoints/final-checkpoint \
  /absolute/path/dataset_manifest.json
```

只有单个完整 8 卡 worker 可用时，使用独立的 8-rank 入口：

```bash
bash scripts/h200/train_a5_fast_state_warmup_8gpu.sh \
  /absolute/path/a2_run/checkpoints/final-checkpoint \
  /absolute/path/dataset_manifest.json
```

它保持 256 个全局 optimizer step、每卡 batch 1（因此 global episode batch 为 8），并把每 rank
DataLoader worker/prefetch 降至 `1/1`，以控制八 rank 的启动期 CPU 与进程压力。

门槛通过后，Main 重新加载同一个 A2 checkpoint，严格叠加 warmup bundle，恢复部分 Qwen
解冻并训练 4 epoch：

```bash
bash scripts/h200/train_a5_associative_lttt_finalonly.sh \
  /absolute/path/a2_run/checkpoints/final-checkpoint \
  /absolute/path/warmup_run/a5_warmup_bundle \
  /absolute/path/dataset_manifest.json
```

启动脚本要求当前用户为 `niujunbo`、至少 4 张可见 GPU（8 卡入口严格要求选中 8 张）、共享盘至少 200 GiB 空闲；它先做
manifest 严格加载和共享盘 safetensors 往返 smoke，再创建唯一 run 目录并按选定拓扑执行 4 或 8 个 rank 的训练。
它不会配置 Mac 的本地代理，也不会写入 dirty 的 LLaMA-Factory checkout。

8-step 对照入口：

```bash
bash scripts/h200/benchmark_fullprefix256_8step.sh baseline
bash scripts/h200/benchmark_fullprefix256_8step.sh a2
bash scripts/h200/benchmark_fullprefix256_8step.sh a5 /absolute/path/a2/checkpoints/final-checkpoint
```

## Checkpoint 与续训

- A2 按既定阶段策略保存完整 Trainer checkpoint。A5 Warmup 不保存完整 checkpoint，只在
  成功完成 256 step 后原子发布 `a5_warmup_bundle/`。A5 Main 禁用周期 checkpoint，结束后先在
  `.final-checkpoint.incomplete` 写入并校验模型、optimizer/scheduler/RNG 和 Trainer state，
  再原子发布为 `final-checkpoint/`，完成态只保留一个 checkpoint。
- 同阶段续训必须新建 run，并显式设置
  `TTT_RESUME_CHECKPOINT=/old/run/checkpoints/checkpoint-N`。入口校验 checkpoint 的 stage 与
  `run_config.json` 一致。
- A2→Warmup 和 A2+handoff→Main 都创建全新的 optimizer/scheduler/RNG；handoff bundle
  绑定 A2 checkpoint、project config、dataset manifest、seed 与代码 commit hash。
  该绑定取 `project.model_dump()` 的 sha256，因此任何 **字段级** config schema 变更都会作废
  已有 bundle。当前 Memory/State Warmup 使用 bundle schema 2：256-step 合同以及冻结 W0/
  slow-projection 的参数组边界使旧的 128-step bundle schema 1 全部失效，必须重跑 Warmup
  重建。只删 validator 不改字段则不影响该哈希。
- `final-checkpoint/` 保存最终模型，`resume_state/` 保存 Accelerator 完整分布式状态；运行中断
  时可从尚存的最后一个标准 `checkpoint-*` 新建 run 续训。
- transient `M`、Bank、FSM、视觉/时序 cache 从所有 checkpoint 中排除。

同阶段续训示例：

```bash
export TTT_RESUME_CHECKPOINT=/absolute/path/old_run/checkpoints/checkpoint-20
bash scripts/h200/launch_4gpu.sh a5
```

## Warmup 释放门（schema-14，bundle schema 2）

256-step Memory/State Warmup 以下列五门判定（发射点：`a5_memory_numerical_audit` trace 与
`memory/*` 指标；空值列为机制未生效时的读数）：

| 门 | 定义 | 阈值 | 空值 |
|---|---|---|---|
| G1 episode 内召回 | `memory/readout_target_cosine`（写前 cos(Mk,v)），episode 末段 vs 首段 | 末−首 ≥ +0.15 且末 ≥ 0.2（最后 32 步均值） | 0（M=0 起点；768 维随机 ≈ 0.029） |
| G2 读取显著性 | `memory/readout_share` = ‖α⊙(K Mᵀ)‖ / ‖f_W0(K)‖（Query 重编码处） | ∈ [0.01, 0.3]，且逐 token 相对效应中位数 > 2⁻⁸ | 0 |
| G3 反事实 | vs `episode_zero`：descent cosine 均值；正增益率 CI（n = interval 2 × 4 rank × 2 query/rank） | cosine ≥ 0.05；95% CI 排除 0.5 | 0 / 50% |
| G4 管路健康 | cotangent 裁剪率 @ max_norm 10；η 重归一率；各组 Outer 裁剪率 | < 30% / < 20% / < 30% | — |
| G5 基础设施 | Qwen bitwise（param+persistent buffer 范围）通过、bundle 发布、≥95% Support 执行写入 | 硬性通过 | — |

次级仪表（无门槛）：`memory/eta_sum`、`memory/beta`、`memory/memory_norm`、
`memory/post_write_cosine`（预期 > 0.8）、probe attention 熵、各 role Query loss
（中位数与均值）、原始 loss 分量（不看 “outer total”）。

## 验收入口

```bash
python -m pytest -q \
  tests/test_fast_ttt.py \
  tests/test_meta_trainer.py \
  tests/test_episode_data.py \
  tests/test_stage_a_targets.py \
  tests/test_production_factory.py

bash -n scripts/h200/launch_4gpu.sh scripts/h200/launch_8gpu.sh
```

CPU 测试覆盖 `T=17/K=8`、两次历史截断、数值连续、旧图断开、`M=0` 前向 bitwise 等于
静态前向（A2 保留回归）、单对写入精确召回、并行写入 slot 顺序不变、Ση 预算重归一、
200 步写入收缩界、写入梯度边界（gate/probe/value/β/token-keys 可达、slot state 不可达）、
256 帧 causal Query、LLaMA-Factory 索引一致性、顺序 Query 梯度等价、manifest 防泄漏、
4/8 rank backward parity 和原子 checkpoint 边界。真实 4/8 卡 8B 验收证据写入各自 run 目录。

## H200 观测工具

以下工具只读取训练日志或设备状态，不修改模型、checkpoint 和训练进程。输出应写入对应的
独立 run 目录，避免覆盖历史实验：

```bash
python scripts/h200/capture_gpu_telemetry.py \
  --output runs/<run_id>/gpu_telemetry_300s.csv \
  --seconds 300

python scripts/h200/bridge_train_log_tensorboard.py \
  --train-log runs/<run_id>/train.log \
  --logdir runs/<run_id>/tensorboard_bridge

python scripts/benchmark_retrieval_history.py \
  --device both \
  --output runs/<run_id>/retrieval_history_benchmark.json
```

GPU 遥测在完成后写同名 `.done` 哨兵。TensorBoard bridge 需要安装项目的 `tracking` extra；
Retrieval benchmark 比较逐行写入与生产 `append_many()` 批量写入当前 tensor ring 的耗时，
不再依赖已删除的 legacy tuple backend。

## Preprocess cache 预热

`scripts/h200/prewarm_preprocess_cache.sh` 是唯一的预热入口，16 路分片写 `runs/<run_id>/`
下的 `command.txt`、`git_state.txt`、`environment.txt`、`shards/` 与 `run_summary.json`。
三条历史配置对应以下调用：

```bash
bash scripts/h200/prewarm_preprocess_cache.sh \
  --stage a2 --roles state_query \
  --cache-root "$PWD/.cache/preprocess/260720_ttt8_benchmark" \
  --cache-namespace 544334e7d7bbf4c2f651 \
  --training-config "$PWD/configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml" \
  --lock-name .state_query_train_prewarm.lock \
  --run-tag svcbench_a2_state_query_cache_train \
  --inspect 1 \
  runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json

bash scripts/h200/prewarm_preprocess_cache.sh \
  --stage a2 --roles "support state_query" \
  --cache-root "$PWD/.cache/preprocess/260723_a2_original_trainsplit_support_statequery" \
  --cache-namespace a2_original_trainsplit_support_statequery_v1 \
  --training-config "$PWD/configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml" \
  --lock-name .support_state_query_train_prewarm.lock \
  --run-tag a2_original_trainsplit_support_state_cache \
  --verify inputs \
  runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json

TTT_H200_VENV=/mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen/.venv-h200-uv-py312-torch28 \
bash scripts/h200/prewarm_preprocess_cache.sh \
  --stage a5 --roles "support state_query" --storage-dtype float16 \
  --cache-root /mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen/.cache/preprocess/260726_a5_support_aligned_v3_fp16 \
  --cache-namespace a5_support_aligned_train_support_statequery_fp16_v3 \
  --training-config "$PWD/configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml" \
  --lock-name .a5_support_state_query_train_prewarm.lock \
  --run-tag a5_train_support_state_cache \
  --verify inputs --inspect 1 \
  runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json
```

`run_summary.json` 为三者的并集 schema：`status`、`stage`、`split`、`roles`、`shard_count`、
`failed_shards`、`shard_exit_codes`、`candidate_chunk_count`、`unique_chunk_count`、
`selected_chunk_count`、`written_bytes` 恒定写出，`verification` 只在 `--verify inputs` 时出现，
`cache` 只在 `--inspect 1` 时出现。任一分片失败、分片 summary 缺失或 verify 非零时
`status=failed` 且脚本退出 1。
