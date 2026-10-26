# MatMul+LeakyReLU 融合分步教程

## 概述

本目录提供 MatMul+LeakyReLU 从单核向量初始版到高性能低阶融合实现的分步教程，共 11 个可执行文件。
整体脉络（4096×4096×4096）：单核向量 6214 ms → 低阶 mix 融合 + L1 四级缓冲 + ASWT 窗口调度 0.81 ms。
融合收益与实现层级相关：高级 API 朴素融合反而更慢（910B 上 mix 模式 C→VECIN 经 GM 中转，
见 Step 2 反例）；低阶 API 自管流水下融合较同配置拆分快约 20%，
LeakyReLU 耗时被重叠进 cube 计算。小 shape 下握手与 launch 开销占比升高，拆分更优
（Step 9/9s 实测小 shape 拆分快约 10~17%）。

Step 1~8 硬编码 M=N=K=4096；Step 9/9s（泛化版）支持运行时传入 `m k n`（矩阵维度，
A[m,k] x B[k,n]），无参运行时执行内置 shape 列表（含大/小/非对齐组合及 `k`/`n` 取 65520 的
跨度边界用例）。参数约束（16 的正整数倍、`k`/`n` 跨度上限等）以 `bash scripts/run.sh --help`
输出为准。

每个 Step 的 `.asc` 内嵌数据生成与精度校验（1024 个随机点对比 double 精度 CPU golden，
输出 `[PASS]`），并打印 `ms/iter` 耗时。
host 侧公共函数（fp16 位转换、随机数据填充、golden 抽检、计时）与多 AIV LeakyReLU kernel
抽取于 `common/` 目录，各 Step 的 `.asc` 仅保留该步的优化点本体。

## Step 列表

| 文件 | 实现要点 | 耗时（4096³） |
|---|---|---|
| `step01_single_core_vector/step01_matmul_single_core_vector.asc` | 单核向量初始版，建立精度基线 | 6214 ms |
| `step02_cube_fused_mix/step02_matmul_cube_fused_mix.asc` | 高级 API mix 融合，因 C→VECIN 经 GM 中转反而更慢（反例） | 1.826 ms |
| `step03_fused_2d/step03_matmul_fused_2d.asc` | 融合 + 二维切分（SetSingleShape 8×4），降低 B 矩阵重复搬运 | 1.524 ms |
| `step04_decoupled_mix/step04_matmul_decoupled_mix.asc` | 解耦融合：cube 写 GM → SyncAll → AIV 读回做 LeakyReLU | 1.411 ms |
| `step05_snake_split/step05_matmul_snake_split.asc` | 高级 API 蛇形窗口调度拆分，通过改善 L2 局部性提升有效带宽 | 1.208 ms |
| `step06_lowlevel_split/step06_matmul_lowlevel_split.asc` | 低阶 API（Nd2Nz/LoadData/Mmad/Fixpipe）手工流水拆分 | 1.227 ms |
| `step06_lowlevel_split/step06b_matmul_lowlevel_split_4buf_aswt.asc` | Step 6 + L1 四级缓冲 + ASWT 4×4 窗口蛇形 | 0.998 ms |
| `step07_fused_lowlevel/step07_matmul_leakyrelu_fused_lowlevel.asc` | 低阶 mix 融合：fixpipe 落 GM 即通知配对 AIV 做 LeakyReLU | 1.005 ms |
| `step08_fused_lowlevel_aswt/step08_matmul_leakyrelu_fused_lowlevel_aswt.asc` | Step 7 + L1 四级缓冲 + ASWT 窗口蛇形，**当前最优** | **0.81 ms** |
| `step09_general/step09_matmul_leakyrelu_fused_general.asc` | Step 8 的任意 shape 泛化（运行时传入 `m k n`，约束见 `run.sh --help`） | 4096³ 回归无损 |
| `step09_general/step09s_matmul_lowlevel_split_general.asc` | Step 9 的拆分对照版（同样支持任意 shape 与命令行 `m k n`） | 拆分对照 |

注：Step 1 为 fp32 输入的单核向量基线；Step 2 起均为 fp16 输入、fp32 累加输出。
跨阶段耗时对比包含输入精度变化，不构成同精度条件下的纯实现优化对比。

## 一键运行（推荐）

仓库提供 `scripts/run.sh`，可一键串联 **构建 → 算子执行 → 结果校验** 全流程。
建议先进入 `fusion_tutorials/` 目录再执行：

```bash
source <CANN_PATH>/set_env.sh
cd Samples/2_Performance/matmul_leakyrelu_fusion_story/fusion_tutorials

# 自动构建 + 未指定 target 时运行 Step 1（step01_matmul_single_core_vector，4096³，需数十秒）
bash scripts/run.sh

# 指定 Step 运行（当前最优）
bash scripts/run.sh --target step08_matmul_leakyrelu_fused_lowlevel_aswt

# 泛化版指定 shape（m k n）
bash scripts/run.sh --target step09_matmul_leakyrelu_fused_general 4160 4080 4048

# 跳过构建阶段
bash scripts/run.sh --target step08_matmul_leakyrelu_fused_lowlevel_aswt --skip-build

# 查看完整帮助
bash scripts/run.sh --help
```

### run.sh 参数说明

完整参数与约束说明以脚本内建帮助为准：

```bash
bash scripts/run.sh --help
```

## 验证与验收标准

- **单步验证**：可执行文件退出码为 0，且首次精度检查与计时后复检（post-timing spot-check）均输出 `[PASS]`。
- **全量回归**：11 个可执行文件全部通过单步验证；其中 Step 9/9s 需无参运行，覆盖内置 shape 列表（大/小/非对齐组合及跨度边界用例共 10 组）。
- **性能复现**：耗时数据与设备型号、CANN 版本强相关。本文耗时在 Ascend 910B4、CANN 9.2.0-beta.2 上实测；计时口径为 warmup 后连续迭代取平均（各 Step 的 warmup/iters 见源码），融合收益以同 Step 的拆分对照版本为准。判定性能结论时应在相同环境下实测对比，而非与本文数值做绝对一致比较。

## 手动构建与运行

如需手动控制各步骤，在仓库根目录执行编译安装：

```bash
cmake -S . -B build -DNPU_ARCH=dav-2201
cmake --build build --parallel
cmake --install build --prefix ./build_out
cd build_out/2_Performance/matmul_leakyrelu_fusion_story/fusion_tutorials
```

随后直接执行某个 Step（示例：Step 8 与泛化版）：

```bash
./step08_matmul_leakyrelu_fused_lowlevel_aswt
./step09_matmul_leakyrelu_fused_general 4160 4080 4048
```

各 `.asc` 也可用 bisheng 单文件直接编译（便于在仓外单独实验；公共头以相对路径包含，需保持
`common/` 与 Step 目录的相对位置）：

```bash
# 高级 API（Step 2~5，需链接 tiling/register；Step 1 纯向量版不需要 -lregister）
bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c <step>.asc -o <step>.o
bisheng <step>.o -o <step> -lm -ldl -lplatform -ltiling_api -lregister
# 低阶 API（Step 6~9）
bisheng <step>.o -o <step> -lm -ldl -lplatform
```

提示：Step 1 编译时加 `-DKK=64` 可缩短 K 维度以快速验证（百毫秒级）。

## 相关文档

- 顶层说明：[`../README.md`](../README.md)
