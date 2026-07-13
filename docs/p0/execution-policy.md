# P0 施工、实验与产物策略

## 每阶段实施前基线

P0 之后，每个 Part 开始前都必须在阶段日志中保存：

- Git branch/commit 与工作树状态；
- `spec_version` 和 `ARCHITECTURE.md` SHA256；
- 配置路径和 SHA256；
- `uv.lock` SHA256；
- 基座模型 ID/revision；
- 数据 fold/split hash；
- 全部随机 seed；
- 上一阶段 `pytest`、Ruff、mypy 结果。

缺少字段必须写 `not-applicable` 或明确阻塞原因，禁止 silent fallback。

## 实验命名规则

固定格式：

~~~text
<spec_version>__fold-<fold>__seed-<seed>__model-<revision12>__ttt-<on|off>
~~~

示例：

~~~text
state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval__fold-train0__seed-42__model-0c351dd01ed8__ttt-on
~~~

不得省略任何字段。重复运行在末尾追加 `__run-<UTC timestamp>`；改变配置、数据划分、模型
revision 或规范 hash 必须生成新 ID，不能覆盖旧产物。

## 产物目录

所有运行产物以环境变量 `OUTPUT_ROOT` 为根，禁止在源码中写死 Windows 或 Linux 绝对路径：

~~~text
${OUTPUT_ROOT}/
└── <experiment_id>/
    ├── manifest.json
    └── p00/ ... p22/
        ├── config/       # 生效配置、路径变量名和 hash；不保存密钥
        ├── logs/         # stdout、stderr、命令和退出码
        ├── checkpoints/ # checkpoint 与 optimizer/fast-state 说明
        ├── metrics/      # 原始及聚合指标
        ├── audit/        # JSON/JSONL：reset、state、retrieval、reader、update
        └── failures/     # 失败样例、traceback、复现命令
~~~

`manifest.json` 至少包含阶段基线的全部字段、硬件/软件版本、开始/结束时间、命令、退出码和
产物 SHA256。checkpoint、大日志和数据不进入 Git；阶段内的小型规范/门禁日志可放在
`docs/pXX/evidence/`。

## 阶段状态：计划设计与已验证实现

“计划设计”只表示 `ARCHITECTURE.md`/`TODO.md` 已定义目标；“已验证实现”必须有当前代码、测试、
日志或实验记录。两列不得互相替代。

| 阶段 | 计划设计 | 已验证实现 |
| :--- | :--- | :--- |
| P0 | 规格冻结与仓库基线已定义 | 已通过：定向 8 tests；全量 13 tests、Ruff、mypy 均绿色 |
| P1 | v5 配置、类型契约与模块骨架已定义 | 已通过：定向 38 tests；全量 47 tests、Ruff、mypy 均绿色 |
| P2 | 数据、预处理、因果切分与 A0 已定义 | 已通过工程门禁：用户批准合成 fold/A0；真实 8B A0 延至 P19/P21/P22 |
| P3 | Qwen 接口、插入点和 DeepStack 保护已定义 | 已通过：定向 56 tests；全量 104 tests、Ruff、mypy、UTF-8 均绿色；tiny/meta 工程证据，真实 8B 留至 P19 |
| P4 | Query/Operator/Time Resolver 已定义 | 已通过：无参位置编码、36.03M Query、9 prototypes、pointer/grammar fail-closed 与本地 tokenizer offset 已验证；模型未训练、阈值留至 P21 |
| P5 | Fast Adapter 与参数收集已定义 | 未实现 |
| P6 | 空间对象编码器已定义 | 未实现 |
| P7 | 时间事件编码器已定义 | 未实现 |
| P8 | 四类 Observation Decoder 已定义 | 未实现 |
| P9 | Semantic Projector、State Bank、FSM 已定义 | 未实现 |
| P10 | Identity Bank 与 Hot Cache 已定义 | 未实现 |
| P11 | Embedding State Retriever 已定义 | 未实现 |
| P12 | Resampler 与 Deterministic Reader 已定义 | 未实现 |
| P13 | Input Composer 与模型编排已定义 | 未实现 |
| P14 | Loss 与 functional SGD 已定义 | 未实现 |
| P15 | Stage A warm-up 已定义 | 未实现/未运行 |
| P16 | Stage B 单步 Meta-TTT 已定义 | 未实现/未运行 |
| P17 | Stage C 一致性与多 support 已定义 | 未实现/未运行 |
| P18 | 测试时协议与推理入口已定义 | 未实现 |
| P19 | 真实 8B/分布式阶段已定义 | 未实现/未运行 |
| P20 | 全量验收与回归契约已定义 | 未实现/未运行 |
| P21 | 消融、校准与待决实验已定义 | 未运行 |
| P22 | clean 评估、审计与发布门禁已定义 | 未运行 |

只有某 Part 的定向验收和当时全部 `pytest`、Ruff、mypy 都通过后，才能把该行更新为已验证，
再勾选 `TODO.md`，随后开始下一 Part。
