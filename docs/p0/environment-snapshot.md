# P0 环境与平台快照

## 本机快照

捕获时间：`2026-07-13T16:35:19.4047380Z`（北京时间 2026-07-14）。原始输出见
`evidence/commands/environment-summary.log`。

| 项目 | 值 |
| :--- | :--- |
| 主机 | `LAPTOP-BS3Q1TRO` |
| 操作系统 | Windows 11 10.0.22631，64 位 |
| CPU | Intel Core i9-13900H |
| 内存 | 33,968,361,472 bytes |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| NVIDIA driver | 596.49 |
| Python | 3.12.13 |
| uv | 0.11.21 |
| PyTorch | 2.9.0+cu128 |
| Transformers | 4.57.1 |
| CUDA runtime | 12.8 |
| CUDA available | true |

## 本机与服务器职责

| 环境 | 固定职责 | P0 已验证状态 |
| :--- | :--- | :--- |
| Windows 本机 | 配置、模块、FSM、loss、functional SGD、reset、静态检查和小张量测试 | 环境与基线命令已验证 |
| Linux 服务器 | 真实 8B 加载、真实视频、FlashAttention、DeepSpeed、多 GPU、正式训练与评估 | 尚未采集；进入 P19 前必须独立快照 |

平台差异必须显式记录：Windows 基线使用 PyTorch CUDA 12.8 wheel，不在基础 lock 中安装
FlashAttention、DeepSpeed 或 bitsandbytes；Linux 服务器需根据其驱动、CUDA、编译器和 GPU
单独验证，若不兼容应使用服务器专用 lock，不能静默修改 `uv.lock` 后沿用同一实验 ID。

## 路径来源契约

运行时只允许从环境变量或路径配置解析资源根目录：

| 用途 | 环境变量 | 配置来源 |
| :--- | :--- | :--- |
| 基座模型 | `QWEN_MODEL_ROOT` | `configs/paths.example.yaml` |
| SVCBench 数据 | `SVCBENCH_ROOT` | `configs/paths.example.yaml` |
| Hugging Face cache | `HF_HOME` | `configs/paths.example.yaml` |
| 输出产物 | `OUTPUT_ROOT` | `configs/paths.example.yaml` |

`.env.example` 中的 Windows 路径只是本机示例，不是源码默认值。P0 对 `src/` 和运行配置进行
绝对路径负向扫描，没有发现 Windows drive 或 `/root`、`/home`、`/mnt`、`/data`、
`/workspace` 硬编码。P1 的配置加载器必须继续执行此契约。
