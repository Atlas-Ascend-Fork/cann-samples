# MatMul+LeakyReLU 融合算子性能优化实践

## 概述

本样例以 `y = leaky_relu(a @ b)` 为例，在 Ascend 910B4（dav-2201）上演示融合算子性能优化的完整过程：从单核向量实现出发，逐轮"观测 → 定位瓶颈 → 优化 → 验证"，在 4096×4096×4096 下从 6214 ms 优化至 1.38 ms（约 99.6 TFLOPS）。

- **完整优化路径**：从高级 API（Cube/mix）到低阶 API（Nd2Nz/LoadData/Mmad/Fixpipe）手工流水，覆盖多级缓冲、ASWT 窗口调度等优化手段
- **融合与拆分对照**：关键节点给出融合与同配置拆分实现的实测耗时对比，量化融合收益与实现层级的关系
- **任意 shape 泛化**：最终版本支持运行时传入 m k n，覆盖大/小/非对齐 shape

## 分步教程

教程位于 `fusion_tutorials/`（9 个可执行文件，每个 Step 一个独立工程）。
各 Step 说明、构建与运行方法、验证与验收标准见 [fusion_tutorials/README.md](./fusion_tutorials/README.md)。
