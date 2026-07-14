# P17 Stage C 身份/事件一致性与多 Support

GATE_STATUS: `passed`

P18_ALLOWED: `true`

ASSET_POLICY: `Synthetic/tiny CPU engineering evidence only; no video, dataset, or 8B weights`

## 已验证闭环

- A4=`pred+0.5*identity`、A5=`A4+0.5*event` 可独立选择，配置差异仅为 event；
- 相邻 chunk O2/E1/E2 snapshot 均 detach+clone、storage 隔离并按位置/时间因果匹配；
- Identity matcher 保留 authoritative hard decision 证据，E1/E2 overlap 分开审计；
- 1/4/8 Support 均为每 chunk 最多一步且 next-only，invalid chunk skip 后时间线继续；
- 多 Query 使用独立 prefill lifecycle、逐点 loss 后求均值；晚 Query 标签不影响早 Query；
- 8-Support backward 后 graph tensor 可回收，连续 episode graph 节点不增长；
- synthetic A4-vs-A3、A5-vs-A4 配对 CI/失败案例报告明确不代表科学增益。

## 证据边界

本门禁仅证明 Stage C 的 CPU 工程因果性、图生命周期和消融工具可执行。runner 尚缺独立
`missed-new` 指标，CPU graph 回收也不等于 GPU 显存曲线；真实增益、阈值和置信结论仍留
P19/P21/P22。

## 验收

P16/P17 联合定向套件 `16 passed`；阶段产物 fail-closed 套件 `8 passed`；Ruff 与 Mypy 通过。
