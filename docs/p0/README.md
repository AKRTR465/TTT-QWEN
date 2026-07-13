# P0 规格冻结与仓库基线

本目录保存 P0 的可审计交付物。P0 只冻结规范、基线和施工规则，不实现或修改模型行为。

## 交付物索引

- [规格锁](./spec-lock.md)：规范身份、基线 hash，以及固定/禁止/实验待定三类边界；
- [环境快照](./environment-snapshot.md)：本机环境、服务器职责、平台差异和路径来源；
- [施工与产物策略](./execution-policy.md)：阶段基线、实验命名、产物目录和计划/实装双栏；
- [需求追踪表](./requirements-traceability.md)：`ARCHITECTURE.md` 各源章节的稳定需求 ID、
  实施阶段和验收位置；
- [仓库基线报告](./baseline-report.md)：DOCUMENT-ONLY 起点、基线命令和 P0 门禁证据；
- `evidence/commands/`：原始命令输出。

所有文本文件使用 UTF-8；仓库中的 Markdown、Python、TOML 和 YAML 由 `.gitattributes`
固定为 LF。
