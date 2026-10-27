# MatMul 分步教程 · arch22（dav-2201 / Ascend910B）

纯 MatMul（`C = A · B`，fp16 输入 / fp16 输出，Cube 内部 fp32 累加）在
**Ascend910B2（dav-2201）** 上的完整性能优化教学样例：6 轮单变量迭代，从"能跑"的
单核向量基线（9922.13 ms），到跑满 cube 流水的最优解（610~637 µs / 215.8~225.4 TFLOPS），
全部真机实测。

## 支持产品

- **仅支持 dav-2201（Ascend910B 系列）**。其他架构（如 dav-3510）下本样例会被
  cmake 自动跳过（`cann_sample_check_arch(dav-2201)`）。

## 目录结构

```
matmul_tutorials/arch22/   #（本目录，位于 matmul_story 下）
├── CMakeLists.txt                    # 仅 dav-2201；每步一个子目录（各自 CMakeLists.txt）
├── README.md
├── scripts/
│   └── run.sh                        # 一键构建 + 依次运行（见下）
├── common/
│   └── host_utils/
│       └── host_utils.h              # 公共 host 工具：错误检查 / fp16<->fp32 位级转换
│                                     # （含次正规数与 ±Inf/NaN）/ golden 抽检 / 计时
├── 1_single_core_vector/             # v1 初始：单核向量（基准）
│   ├── CMakeLists.txt
│   └── step1_matmul_single_core_vector.asc
├── 2_cube_multicore/                 # v2 换 Cube + 多核（高级 API）
│   ├── CMakeLists.txt
│   └── step2_matmul_cube_multicore.asc
├── 3_snake_schedule/                 # v3 蛇形窗口喂数调度
│   ├── CMakeLists.txt
│   └── step3_matmul_snake_schedule.asc
├── 4_stagger_schedule/               # v4 错位分核单开（v3 基座仅换核间映射）
│   ├── CMakeLists.txt
│   └── step4_matmul_stagger_schedule.asc
├── 5_lowlevel_pipeline/              # v5 低阶 API 逐 tile 流水
│   ├── CMakeLists.txt
│   └── step5_matmul_lowlevel_pipeline.asc
├── 6_lowlevel_4buf_stagger/          # v6 = v5 + L1 四级缓冲 + 错位分核（最优）
│   ├── CMakeLists.txt
│   └── step6_matmul_lowlevel_4buf_stagger.asc
├── 7_general_shape/                  # v7 = v6 任意 shape 泛化（M/N/K 为 16 倍数）
│   ├── CMakeLists.txt
│   └── step7_matmul_general_shape.asc
└── 8_full_load/                      # 专题：全载模板（M=32 解码形态）
    ├── CMakeLists.txt
    └── step8_matmul_full_load.asc
```

## 版本与实测数据（M=N=K=4096，fp16 输入 / fp16 输出）

| 文件 | 版本 | kernel 形态 | mm 单项耗时 | 算力 |
|---|---|---|---|---|
| step1_matmul_single_core_vector.asc | v1 初始：单核向量（fp16 进出） | 1×AIV | 9922.13 ms | 0.014 TFLOPS |
| step2_matmul_cube_multicore.asc | v2 换 Cube + 多核（高级 API，1D 切分） | 16×AIC | 785~793 µs | ~174 TFLOPS |
| step3_matmul_snake_schedule.asc | v3 蛇形窗口喂数调度（4×4 窗口，高级 API） | 16×AIC | 789 µs（与 v2 持平） | 174.1 TFLOPS |
| step4_matmul_stagger_schedule.asc | v4 错位分核单开（高级 API，v3 基座仅换核间映射） | 16×AIC | 780 µs（-1.1%） | 176.3 TFLOPS |
| step5_matmul_lowlevel_pipeline.asc | v5 低阶 API 逐 tile 流水（Nd2Nz/LoadData/Mmad/Fixpipe + 双缓冲旗环） | 16×AIC | 653 µs（-17%） | ~210 TFLOPS |
| step6_matmul_lowlevel_4buf_stagger.asc | **v6 = v5 + L1 四级缓冲 + 错位分核（最优）** | 16×AIC | **610~637 µs** | **215.8~225.4 TFLOPS** |
| step7_matmul_general_shape.asc | v7 = v6 任意 shape 泛化（M/N/K 为 16 倍数，运行时传入） | 16×AIC | 4096³ 回归 616.6 µs | 222.9 TFLOPS |

> 口径说明：上表为 **Ascend910B2（dav-2201）+ CANN 9.3.0、2026-10 同环境实测**（v6 为多次
> 测量区间）。计时为 host `steady_clock` 均值（预热 ≥5 次 + 迭代 ≥20 次），**含 kernel launch
> 与流同步开销**（本 shape 下为 µs 级），并非纯 kernel 耗时；每组数据另以 msprof 的 kernel
> Task Duration 复核，双口径差 <2% 互认。dtype：fp16 输入 / fp16 输出（Cube 内部 fp32 累加）。

结论：**在上述实测环境（910B2 + CANN 9.3.0）、4096³ fp16 用例与本教程候选实现范围内，
纯 MatMul 的最优是 step6（610~637 µs / 215.8~225.4 TFLOPS）**。
收益结构：

1. **选对计算单元 + 多核**：向量单元做矩阵乘只能一行行 Axpy 累加；
   Cube 一拍 16×16×16 MAC 阵列。工具错配时，任何微优化都救不回来。
2. **喂数调度（v2 → v3）**：流量下界 = M·N·K·2B·(1/baseM+1/baseN) = 512 tile ×（A 条 1 MiB +
   B 条 2 MiB）= 1.5 GiB，被 L0C=128KB 钉死（base tile ≤128×256）；蛇形窗口
   留作带宽受限平台的通用工具。
3. **错位分核单开（v3 → v4）**：每拍 16 核从 4×4 窗口（M/N 各 4 条带、4 核/条带）改为
   16 条 M × 16 条 N 对角线落点（各 1 核），同侧 bank/通道冲突核数 4→1（对朴素序 16→1）。
4. **低阶 API（v3 → v5）**：手工通路 GM→L1(ND2NZ)→L0(LoadData)→L0C(Mmad)→GM(Fixpipe)，
   双缓冲事件旗环 + 尾部排空 ，调度自由度直接兑现为吞吐。
5. **L1 四级缓冲 + 错位分核（v5 → v6）**：L1 环 2→4 级；错位分核（迭代③同款映射，低阶 32×16 网格）
   让每拍 16 核在 M/N 两向错开（同侧 bank/通道冲突核数 16→1）。
6. **泛化（v7）**：见下方泛化小节。

v6 的调参负结果：5 级缓冲持平（深度饱和）、K_L1=256 回退、
unitFlag=2/3 无变化（单 L0C 累加器下 final accumulation 都在最后一发 mmad）、FIX_M 等待下沉
无变化（fixpipe 占比低）、L0C 双累加器（2×128×128）回退 ~15%（mmad/LoadData 指令数翻倍）、
窗口 2×8 持平 / 8×2 回退（4×4 的 A/B 复用最均衡）。生产实践调研（冲突类优化）的正面收获：
错位分核块分配公式（已吸收进 step6，并在 step4 单开验证）。

## 编译运行

### 方式一：仓内 cmake（推荐）

```bash
# 仓库根目录执行（需已 source CANN 环境），逐行可连续执行。
# 注意：cmake --install
cmake -S . -B build -DNPU_ARCH=dav-2201
cmake --build build --parallel     # 全量构建
cmake --install build --prefix ./build_out     # 产物在 build_out/2_Performance/matmul_story/matmul_tutorials/arch22/
(cd build_out/2_Performance/matmul_story/matmul_tutorials/arch22 && ./step6_matmul_lowlevel_4buf_stagger)   # 子 shell 运行，当前目录保持在仓库根

# —— 单目标快速验证：只构建该目标、不执行 install，直接运行构建树内产物 ——
cmake --build build --target step4_matmul_stagger_schedule
./build/Samples/2_Performance/matmul_story/matmul_tutorials/arch22/4_stagger_schedule/step4_matmul_stagger_schedule

# —— 一键构建并依次运行 step2~8（跳过慢速 step1，见脚本注释）——
bash Samples/2_Performance/matmul_story/matmul_tutorials/arch22/scripts/run.sh
```

### 方式二：bisheng 单文件直编（不依赖仓内构建）

```bash
source <CANN_PATH>/set_env.sh

# step1（单核向量版，不需要 -lregister）
bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 1_single_core_vector/step1_matmul_single_core_vector.asc -o step1.o
bisheng step1.o -o step1 -lm -ldl -lplatform -ltiling_api
./step1            # 4096³ 单核约 30s（很慢，属正常）；加 -DKK=64 可快速跑通

# step2 / step3 / step4（用了 MultiCoreMatmulTiling，需要 -lregister）
bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 2_cube_multicore/step2_matmul_cube_multicore.asc -o step2.o
bisheng step2.o -o step2 -lm -ldl -lplatform -ltiling_api -lregister
./step2

bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 3_snake_schedule/step3_matmul_snake_schedule.asc -o step3.o
bisheng step3.o -o step3 -lm -ldl -lplatform -ltiling_api -lregister
./step3

bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 4_stagger_schedule/step4_matmul_stagger_schedule.asc -o step4.o
bisheng step4.o -o step4 -lm -ldl -lplatform -ltiling_api -lregister
./step4           # 错位分核单开（v3 基座只换核间映射）：实测 780 µs / 176.3 TFLOPS

# step5 / step6 / step7（低阶 API，不需要 -lregister）
bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 5_lowlevel_pipeline/step5_matmul_lowlevel_pipeline.asc -o step5.o
bisheng step5.o -o step5 -lm -ldl -lplatform
./step5

bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 6_lowlevel_4buf_stagger/step6_matmul_lowlevel_4buf_stagger.asc -o step6.o
bisheng step6.o -o step6 -lm -ldl -lplatform
./step6

bisheng -std=c++17 -fPIC --npu-arch=dav-2201 --asc-aicore-lang -c 7_general_shape/step7_matmul_general_shape.asc -o step7.o
bisheng step7.o -o step7 -lm -ldl -lplatform
./step7           # 跑内置 10 组 shape；或 ./step7 <m> <k> <n> 单跑（如 ./step7 4160 4048 4080）
```

正确性校验已内嵌：各 Step 对 1024 个固定种子随机位置做 double 精度 CPU golden 抽检，
**首次执行与计时结束后各检一次**（连续运行后的输出重新拷回复检，两次结果共同决定退出码；
fp16 解码后非有限值直接判失败）。输出含两行 `[PASS]`/`[FAILED]` 与一行 `ms/iter`。

## 任意 shape 泛化（step7）

step7 = step6 的泛化版：命令行 `./step7 <m> <k> <n>`（参数顺序与 arch35 教程统一为
m k n；无参则跑内置列表）。约束：**m/n/k 均为 16 的正整数倍**（fp16 32B 突发 = 16 元素，
GM 行距因此 32B 对齐），且 **k、n ≤ 65520**（Nd2Nz 搬运的 GM 行距字段 srcDValue 上限为
65535，取 16 对齐最大值 65520；超出直接报错退出，如 `./step7 16 65536 16`）。
机制：tile 网格 ceil 划分 + 错位分核映射（各核独立计算块号，尾波空位跳过）；边缘 tile 按
min(128/256, 剩余) 实际尺寸走全通路、免零填充。内置 shape 列表覆盖非 tile 对齐、K 尾块、
小 shape 与 **k/n=65520 上限边界用例**（16×16×65520、16×65520×16，另含 65536 拒绝路径），
逐组 golden 校验；4096³ 回归 616.6 µs（与 step6 区间一致）。

## 专题：全载模板（step8，M=32 解码形态）

场景：一路输入显著小的 MatMul（增量/解码：M=32、K=N=4096，fp16 进出）。A（32×4096 = 256KB）
可完整驻留 L1，但通用模板会**按 N-tile 重复搬运 A**。`step8_matmul_full_load.asc` 在同一文件内
放两个 kernel 做单变量对照（B 环、L0 双缓冲、事件旗环、Fixpipe 完全相同，只有 A 的搬运策略不同）：

- `mm_stream`：通用模板 —— A 随 K-chunk 进 L1 环，每个 N-tile 整条重扫
- `mm_fullload`：aL1 常驻 —— A 一次 Nd2Nz 全载（`[32×4096]` NZ = 256KB），K 循环只搬 B，
  A 的 K 窗口用 LoadData 分形偏移原地寻址（`k1 = kc*(K_L1/16) + j*(K_L0/16)`）

- 收益 ≈ A 尺寸 ×（每核 N-tile 数 − 1）：N=8192（每核 4 tile）时总流量 80 → 68 MiB（-15%）
- A 从"每 N-tile 重扫"，变为"驻留一次"的直接兑现
