## 变更范围

- 当前阶段：P__
- 直接依赖阶段已通过：
- 对应需求 ID：
- 计划设计变更：
- 已验证实现变更：

## 固定契约

- [ ] 变更与 `ARCHITECTURE.md` 当前规范名和 hash 一致；如不一致已停止施工并先更新追踪表。
- [ ] 未把旧 v3 YAML、历史测试或 `__pycache__` 当成 v5 已实现证据。
- [ ] shape、mask、dtype、device、梯度/更新边界、reset/隔离和 query-time 因果性均有证据。
- [ ] 模型、数据和输出路径只来自环境变量或路径配置，源码没有平台绝对路径。
- [ ] 文档、配置、实现、契约测试和阶段日志在同一阶段同步。

## 第一版禁止项

- [ ] 未引入 Surprise Gate 或学习型更新 Gate。
- [ ] 未引入 Inner AdamW、Muon、momentum SGD 或每 chunk 多步更新。
- [ ] 未让 LLM 或连续 embedding 直接回归最终累计整数。
- [ ] 未引入固定 Top-K 状态检索。
- [ ] 未给 O1 添加无标签一致性 loss。
- [ ] 未改造 DeepStack。
- [ ] 未启用 ANN 向量数据库或堆叠额外 online-update 正则。
- [ ] 未使用 query_time 之后的帧或任何答案/计数标签字段。
- [ ] 未在 generate 自回归循环中重复执行 Bank 或 TTT 更新。

## 实验与验收

- [ ] 实验 ID 包含规范版本、数据折、seed、模型 revision 和 TTT 开关。
- [ ] 阈值或结构选择只使用训练折/独立校准集，并记录完整搜索空间和 provenance。
- [ ] 已先跑本阶段定向测试，再跑当前全部 `pytest`、`ruff` 和 `mypy`。
- [ ] 原始命令日志、配置快照、指标、审计 JSON 和失败样例已写入阶段产物目录。
- [ ] 若本阶段通过，`TODO.md` 和实现状态表只在全部门禁绿色后标记完成。
