# P3 Qwen3-VL 接口、Merger 插入点与 DeepStack 保护

GATE_STATUS: `passed`

P4_ALLOWED: `true`

ASSET_POLICY: `no video or 8B-weight download`

VALIDATION_SCOPE: `official HF modules on meta + tiny random-weight HF model`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `b2d6f8d647860eac9a3bb4f7ed01acfccebbd4f3`（P2 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `0690c9cf5d8301b644abd87deb01a0a02b3126e30eaa3d18e69b5bb105c57adc` |
| config SHA256 | `f05e8430532e8f7eb32421091fee2802d663fdd19c92e649641a5795dd8bdb9a` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| Transformers source SHA256 | `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| dataset/fold | `not-applicable`（P3 不读取数据） |
| tiny model seed | `0` |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已实现且有本地证据的部分

| 范围 | 已验证实现 |
| :--- | :--- |
| 轻量 fail-fast loader | 先 local-only 读取 config 并断言 27/1152/16、patch/merge、4096、8/16/24、36/4096；失败时不触碰权重 loader，通过后权重加载仍强制 local-only |
| 官方 shape | meta device 验证 `[1568,1536]→[1568,1152]`、全部 27 blocks shape 不变、group dim 4608、Main `[392,4096]`，boundary 严格暴露 `[1,392,4096]` 和 `[8,14,14]→[8,7,7]` |
| 真实插入点 | 临时包装内层 `model.get_video_features()`，事件顺序为 Main Merger→Adapter→video `masked_scatter`；不 hook image 共用 merger |
| batch mapping | 原生 packed `[sum N_i,4096]` 与 per-video split 保留；额外暴露 left-aligned padding view、valid mask、merged grid、token counts/offsets |
| DeepStack | 三组原 tensor 不经 Adapter/Bank/新 mask，保持对象、顺序、dtype/device 和 mask，按原生语义在 decoder 0/1/2 后注入 |
| 等价与隔离 | disabled 时 Main、DeepStack、video/image/text logits bitwise 等价；enabled 时 image-only/text-only 不调用 Adapter，mixed 输入只变换 video Main |
| 梯度与冻结 | Qwen 默认 `requires_grad=False` 且持续 eval；不使用 `no_grad` 包裹 forward，Adapter 梯度非零且有限 |
| 生命周期 | forward/generate/direct 调用共用互斥锁；禁止重入和双适配；异常后恢复原方法并清除 stale capture；generate 只在 prefill 调用一次 Adapter |
| PyTorch 契约 | inner owner 不重复注册；`state_dict` 无 `feature_owner.*`；业务方法不覆盖 `nn.Module.apply()` |

P2 processor 的单样本输出可带 `[1,N,1536]` batch 维；进入本边界前必须通过
`flatten_for_qwen()`，多样本再按 Qwen 原生约定拼成 `[sum(N_patch),1536]`。P3 `VideoBatch`
显式拒绝 3D、空 batch、非正 grid 和 packed patch 数不一致，防止 P13 误接接口。

## 真实调用链

~~~text
PatchEmbed
→ 27 Vision blocks
→ DeepStack mergers at ViT 8/16/24
→ Main merger
→ inner Qwen3VLModel.get_video_features()
→ cat video embeddings
→ video masked_scatter
→ decoder layer 0/1/2 后依次注入 DeepStack 0/1/2
~~~

外层 `Qwen3VLForConditionalGeneration.get_video_features()` 不在真实 forward 路径；直接 hook
`visual.merger` 会同时命中 image，因此本实现不采用这两个位置。

## 证据边界

本阶段没有下载模型、视频或额外数据。官方维度只在 meta device 验证；端到端 forward、梯度、
mask、异常和 generation 使用 tiny 随机权重 Qwen3-VL。它们完成 P3 工程契约，但不能称为真实
Qwen3-VL-8B 集成结果，也不能支持精度、吞吐或显存结论。真实 8B 加载、hook、DeepStack、
device/dtype 和分布式行为必须在 P19 重跑；P5 负责真实 Fast Adapter 的参数与 device/dtype 放置。

## 验收结果

验收时间：`2026-07-13T19:13:03.7685335Z`。强制离线环境下，P3 定向验收为 56 passed；
全量 `pytest` 为 104 passed；`ruff check .`、`mypy src` 和严格 UTF-8 解码均通过。

## 证据索引

- `evidence/commands/p3-baseline.log`
- `evidence/commands/p3-qwen-call-chain-audit.log`
- `evidence/commands/p3-targeted-pytest.log`
- `evidence/commands/p3-full-checks.log`
- `evidence/commands/p3-utf8-audit.log`
