# 当前实施进展与服务器迁移说明

更新日期：2026-07-14

## 1. 当前结论

项目目前处于“本地工程闭环已基本建立，真实 8B 集成与科学验证待服务器完成”的阶段：

- P0–P17 已通过 synthetic/tiny/小张量 CPU 工程门禁。
- P18 已完成测试时 runtime 协议骨架，但尚不是可直接用于真实 Qwen3-VL-8B 的完整生产推理闭环。
- P19–P22 的真实 8B、分布式训练、阈值校准、消融和正式评估尚未开始，应在服务器端完成。

当前结果证明的是代码接口、状态隔离、因果顺序、梯度路径和失败边界等工程合同；没有加载真实 8B 权重，也没有使用完整 SVCBench 视频，因此不能据此声明训练收敛、真实精度提升或 State-TTT 的科学增益。

## 2. 已完成的关键阶段

### P0–P15：基础模块与 Stage A

配置、数据合同、Qwen 接入边界、Query/Operator/Time Window、Fast Adapter、空间/时间状态编码、四类 Observation Reader、Bank/FSM、Retriever、模型编排、typed loss、functional SGD 以及 Stage A 训练编排均已完成对应的 synthetic/tiny CPU 工程验证。

### P16：Stage B 单步 Meta-TTT

P16 已完成 A3 工程闭环，核心路径为：

`Support observe → 无标签 L_pred → functional SGD → 后续 Query → outer loss/outer gradient`

已验证单 Support、每个有效视频行至多一步更新、更新只影响后续 Query、full-second-order 梯度能够回到初始 fast weights，以及 reset、skip、状态隔离和标签防泄漏边界。

### P17：Stage C 多 Support Meta-TTT

P17 已完成 synthetic/tiny CPU 工程闭环：

- A4 显式启用 `L_pred + L_id`，A5 显式启用 `L_pred + L_id + L_event`；
- 支持 1/4/8 个连续 Support 和多个后续 Query；
- 覆盖 O2/E1/E2 的因果 overlap matching、detached snapshot 和跨 chunk runtime 编排；
- 已验证 bounded CPU graph 回收、outer gradient、next-only 更新与标签隔离。

仍未完成的 P17 证据有两项：独立 `missed-new identity` runner 指标尚未接入；CPU graph 回收不能替代真实 GPU 峰值显存与长期运行稳定性证据。

### P18：测试时 runtime 骨架

P18 已完成并验证以下协议骨架：

- 每视频原子 reset、checksum 和 release；
- 按 query time 的因果裁剪，拒绝未来信息进入更新或回答；
- Fast Adapter runtime state 的受管绑定；
- Stage A runtime bridge 和可注入 update stage；
- 当前 chunk 更新只影响下一 chunk；
- Query 阶段单次 read/compose/prefill；
- decode 阶段不修改 Bank、fast weights 或其他 runtime；
- 中止或异常后安全释放，避免污染下一视频。

P18 尚缺两项生产接线：真实 `L_TTT → functional SGD` updater 尚未接入 `PerVideoRuntimeManager`；`GenerationDriver` 尚未在真实 Qwen3-VL-8B 上完成 generation。因此，当前 P18 应称为 runtime 骨架，不应称为正式 8B 推理完成。

## 3. 当前验证结果

本地收束验证结果如下：

- `pytest -q`：617 passed；
- Ruff：通过；
- Mypy：通过；
- 配置 CLI 与推理协议 CLI：通过；
- 本轮变更文件格式检查：通过；
- `git diff --check`：通过。

这些检查均属于 CPU-safe 工程验证，不包含真实 8B 权重加载、GPU 混合精度、吞吐、显存、收敛或正式指标。

## 4. 服务器端建议实施顺序

1. **建立可复现环境**：在服务器重新创建 Python/CUDA/PyTorch/Transformers 环境，记录版本和安装命令；不要复制本地虚拟环境。
2. **准备外部资产**：在仓库外配置 Qwen3-VL-8B checkpoint、SVCBench 视频/标注、Hugging Face 缓存和实验输出目录，并锁定 checkpoint revision 与 hash。
3. **先复跑 CPU 门禁**：在服务器新环境运行完整测试、Ruff、Mypy 和两个 CLI smoke test，确认迁移没有改变现有合同。
4. **完成 P18 生产接线**：把真实 typed `L_TTT` 和 `functional_sgd_steps_from_ttt` 接入 update stage，并用 tiny/单视频先验证 reset、next-only、异常释放和 decode immutable。
5. **进入 P19 最小 8B 集成**：先完成真实 Qwen3-VL-8B 单卡或最小可行配置的一次端到端 episode，验证 hook、dtype/device、BF16 数值稳定性、真实 generation 和显存上限。
6. **再启用性能组件**：在最小 8B 路径稳定后接入 FlashAttention、DeepSpeed 和多 GPU，补充吞吐、峰值显存、并发及中断恢复证据。
7. **完成 P20–P22**：固化全量回归合同，进行阈值校准、A0–A5 公平消融、clean split 正式评估，并补齐独立 `missed-new identity` 等离线指标。

服务器推进时应先取得“单视频、单 episode、可复现、可审计”的最小真实 8B 证据，再扩大训练规模；synthetic/tiny 结果与真实 8B 实验结果需在日志和报告中明确分栏。

## 5. 代码同步与资产边界

远端仓库只同步源码、配置、测试和文档。以下环境或大体积/运行时资产不应提交：

- `.venv/`、Conda 环境目录、Python 缓存、IDE 本地配置；
- Qwen/SVCBench 权重、数据集、视频、Hugging Face 缓存；
- checkpoint、optimizer/runtime snapshot、训练日志、临时输出和性能 profile；
- API key、访问令牌、服务器地址等任何密钥或私有配置。

服务器端应通过环境文件、启动参数或未跟踪的本地配置提供资产路径。正式实验需另行记录 commit、配置、seed、checkpoint revision/hash、CUDA/依赖版本、启动命令和输出位置，以保证结果可追溯。

## 6. 下一阶段完成标准

只有在服务器端完成真实 Qwen3-VL-8B 权重加载、生产 updater 接线、真实 generation、至少一个端到端 episode，以及 GPU 显存/数值/状态隔离审计后，才能把 P18/P19 标记为真实 8B 集成完成。训练收敛、阈值选择和效果增益仍需 P21/P22 的正式实验单独证明。
