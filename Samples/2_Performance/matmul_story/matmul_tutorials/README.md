# MatMul 分步教程（按架构组织）

本目录按 NPU 架构组织分步教程，每个架构一个子目录：

| 子目录 | 架构 | 内容 |
|---|---|---|
| [arch35](./arch35) | dav-3510（Ascend950 系列） | MXFP4 量化矩阵乘分步教程：`0_naive` → `7_fullload` 共 8 个 Step，配套 `scripts/run.sh` 一键构建/运行/校验 |
| [arch22](./arch22) | dav-2201（Ascend910B 系列） | 纯 MatMul（fp16 进出）6 轮单变量优化教学：`1_single_core_vector` → `8_full_load`（单核向量→Cube 多核→蛇形窗口→错位分核→低阶流水→4buf+错位） |

各架构的编译、安装与运行说明见对应子目录的 `README.md`（如 [arch35/README.md](./arch35/README.md)）。
