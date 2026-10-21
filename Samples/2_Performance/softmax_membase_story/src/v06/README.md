# v06：两遍读取的 OnlineSoftmax

[总览](../../README.md) · [上一版：v05](../v05/README.md)

v06 还是只改长行的 `GENERAL` 路径，这次换算法：三遍流程改成两遍。三遍流程慢在要先看完整行才能定最大值，分子不得不在 GM 里暂存一轮。OnlineSoftmax 让最大值和分母一起滚动更新，第一遍读完输入就有了整行统计量，第二遍重读输入直接算最终结果。分子不再出 UB 以外的暂存，每元素搬运请求从 20 字节降到 12 字节。

## 在线更新最大值与分母

![v06 在线更新最大值与分母的两遍流程](../../images/alg/softmax_online.png)

第一遍只维护 UB 里的行状态，不写 GM；第二遍用最终的 m 和 s 重读输入，直接写出结果。图中箭头表示数据依赖，不按真实时间缩放。[可编辑图源](../../images/alg/softmax_online.excalidraw)。

设输入 $\mathbf{X}\in\mathbb{R}^{M\times N}$，$M=\mathrm{rows}$、$N=\mathrm{cols}$。对第 $i$ 行，$m_i$、$s_i$ 是已处理部分的最大值和分母，$\mathcal{J}$ 是当前 tile 的有效列。每读入一个 tile，按三行更新（撇号表示新状态）：

$$
\begin{aligned}
m_i' &= \max\left(m_i,\max_{j\in\mathcal{J}}X_{ij}\right),\\
\alpha_i &= \exp(m_i-m_i'),\\
s_i' &= \alpha_i s_i+\sum_{j\in\mathcal{J}}\exp(X_{ij}-m_i').
\end{aligned}
$$

关键在 $\alpha_i$：新 tile 抬高了最大值时，旧分母要乘上 $\exp(m_i-m_i')$ 换算到新基准，再加上当前 tile 的贡献。每轮用 $m_i'$、$s_i'$ 覆盖旧状态。首个 tile 没有旧状态，直接用本块的最大值和指数和初始化。整行处理完后，第二遍重读输入，算 $Y_{ij}=\exp(X_{ij}-m_i)/s_i$。

### 分块计算示例

输入 `[0,1,2,3,4]`，每段两个元素（示例段长只为说明，实际完整 tile 按 64 个 FP32 对齐）：

| 输入段 | 新最大值 | 更新后的分母 |
|---|---:|---|
| `[0,1]` | 1 | `exp(-1)+1`，首段直接初始化 |
| `[2,3]` | 3 | `旧s*exp(-2)+exp(-1)+1` |
| `[4]` | 4 | `旧s*exp(-1)+1` |

最终 `m=4`，`s=exp(-4)+exp(-3)+exp(-2)+exp(-1)+1`，正好是以 4 为基准的整行分母；第二遍输出 `[exp(-4),exp(-3),exp(-2),exp(-1),1]/s`。

注意 v06 每个 tile 都立即做最大值归约和求和归约，没有沿用 v05 的“逐位置累计、行末归约”。两版是长行的两种不同改法，不是叠加关系。

## 三个目标 shape 的计算路径

| shape | 路径 | 本版如何计算 |
|---|---|---|
| `(1024,1024)` | `FULLLOAD_WITH_DB` | 与 v04 相同：整行留 UB，输入输出共用缓冲，两行交替 |
| `(1024,16384)` | `FULLLOAD_WITH_DB` | 与 v04 相同 |
| `(128,131072)` | `GENERAL` | 第一遍在线更新最大值和分母，第二遍重读输入写结果 |

整行单缓冲（`FULLLOAD`）仍用基础归约和 `GetValue`，`cols=1` 仍直接填 1，本版没动这两条路径。

## 在线状态留在向量计算中

`mLocal_`、`sLocal_`、`alphaLocal_` 各占 32B，里面存 8 份相同的值。归约走 v03 的流程：高级 Reduce 沿 repeat 方向归约，`WholeReduce` 合成一个值，`Brcb` 广播进首 32B。后续向量减法、乘法用零源步长反复读这个块，全程不用 `GetValue` 取标量。

更新分母用一条融合乘加：

```text
FusedMulAdd(sLocal_, alphaLocal_, tileReducedValueLocal_, 8)
即：sLocal_ = sLocal_ * alphaLocal_ + tileReducedValueLocal_
                旧s       缩放因子          当前 tileSum
```

一条指令完成分母递推里的乘和加。计算之间的依赖仍要等向量运算完成；双缓冲只让独立的搬运和计算重叠，不会消除算法自身的依赖。

## UB 与全局内存

两份 `xy` 在搬入、计算、搬出之间交替，输入在各自缓冲内原地变成输出。`m/s/alpha` 和归约工作区各一份。

设 `B=align256(tileCols*4)`，`Q=reduceTmpBytes`：

| UB 区域 | 份数与容量 | 用途 |
|---|---|---|
| `xyLocal` | 两份，每份 B | 输入原地计算，交替搬运 |
| `tmpLocal_` 前区 | 一份，Q | 高级归约工作空间 |
| `tmpLocal_` 尾区 | 一份，256B | 先存新最大值，再复用为 tileSum；有效结果在首 32B |
| `mLocal_`、`sLocal_`、`alphaLocal_`、`onesLocal_` | 各 32B | 最大值、分母及倒数、缩放因子、常量 1 |

总量 `2B + Q + 384` 字节。host 查询高级 Max/Sum 归约的最大临时空间，取较大者按 32B 对齐，再二分查找 UB 能容纳的最大完整 repeat 数。

可用 UB 196352B、CANN 9.1.1 下，GENERAL 选 16384 列一段，`Q=32768B`，共申请 164224B。131072 列恰好分 8 段。没有累计区，所以 tile 比 v05 的 13568 列大。

按一行 N 个 FP32 计，三种流程的搬运请求量：

```text
三遍（v04/v05）：读x + 读x + 写分子到y + 读y + 写y = 20N 字节
两遍（v06）：    读x + 读x + 写y                    = 12N 字节
```

这是算法发出的请求量，不是实测 DRAM 流量。代价是指数从算一遍变成算两遍，外加每 tile 一次的在线状态更新——用计算量换搬运量。

第一遍虽然不写 GM，仍保留队列从计算到搬出的完成交接才释放输入缓冲，避免下一次搬入覆盖还没读完的数据。

## 任意列数与尾块

支持任意正整数 `cols`。缓冲容量按 256B 向上对齐，有效计算长度用 `currTileCols`：

| 最后一个 tile 的区域 | Max 归约前 | Sum 归约前 |
|---|---|---|
| 前 `currTileCols` 个有效位置 | 保留输入 | 保留有效指数 |
| 最后一个 repeat 的无效位置 | 填 `-Inf` | 填 0 |

拷入用 `DataCopyPad`，第一遍把不足 32B 的尾块补 `-Inf`。`ReduceToFirstBlock` 在归约前填满最后一个 repeat。指数和最终乘法只处理有效元素，写回只写有效字节，不覆盖下一行。最后一个 tile 较短时，在线递推就按这段的实际数据更新。

整行双缓冲用相同的归约尾部填充；整行单缓冲的基础归约按有效列数计算。输入要求有限 FP32，不处理 NaN/Inf；Max 尾部的 `-Inf` 是内部填充值。

## 按完整行分核

与前版相同：核数取请求核数与行数的较小值，前 `rows%cores` 个核各多一行，`Init` 把地址偏移到本核首行。一行内的在线状态只由一个核维护，不用跨核合并。

40 核处理 1024 行，前 24 核各 26 行、其余各 25 行；处理 128 行，前 8 核各 4 行、其余各 3 行。`cols=1` 按元素均分，每核平均至少 64 个 FP32，不足时只开一核。

## 代码结构与执行流程

各文件按计算路径组织，入口和容量计算在 [softmax.asc](softmax.asc)，kernel 类的 `Init` 分配缓冲并定位本核地址，`Process` 遍历本核的行。

| 文件 | 职责 |
|---|---|
| [softmax.asc](softmax.asc) | 选择 TilingKey，计算分核和 UB 容量，启动对应模板 |
| [softmax_common.h](softmax_common.h) | TilingData、对齐工具、尾部填充和 `ReduceToFirstBlock` |
| [softmax_single_tail.h](softmax_single_tail.h) | `cols=1`，直接填 1 |
| [softmax_full_load_with_db.h](softmax_full_load_with_db.h) | 整行驻留、双缓冲，向量指令完成归约后的计算 |
| [softmax_full_load.h](softmax_full_load.h) | 整行驻留、单缓冲，保留基础归约和标量取值 |
| [softmax_general.h](softmax_general.h) | 长行两遍计算，本版的主要改动 |

模板选择逻辑与 v04 相同；除 GENERAL 外计算都与 v04 相同：

```python
if cols == 1:
    y = 1
elif two_rows_and_workspace_fit_ub:
    FullLoadWithDB.Init(x, y)
    FullLoadWithDB.Process()   # 每行 m=max(x); p=exp(x-m); s=sum(p); y=p/s
elif one_row_and_workspace_fit_ub:
    FullLoad.Init(x, y)
    FullLoad.Process()         # 同一公式；单缓冲，归约后用 GetValue 取标量
else:
    General.Init(x, y)
    General.Process()          # 下方展开
```

下面的 Python 风格伪代码概括 GENERAL 的控制结构和数据依赖。`mLocal`、`sLocal` 等始终是 UB 内的向量状态，`Broadcast` 表示向量指令重复读首 32B，没有标量读取。省略队列申请、释放和屏障，真实实现以头文件为准。

```python
def Init(x, y):
    # 按完整行分核，xGlobal/yGlobal 指向本核第一行
    LocateCoreRows(x, y, tiling)
    Allocate(xy_buffers=2, tmp_buffers=1)
    # mLocal、sLocal、alphaLocal、onesLocal 各 32B
    onesLocal[:] = 1


def Process():
    for row in range(rowsCurrCore):
        Stage1WithOnlineStats(row)
        Stage2(row)


def Stage1WithOnlineStats(row):
    mLocal[:8] = -Inf
    for offset in range(0, cols, tileCols):
        n = min(tileCols, cols - offset)
        xyLocal = CopyIn(xGlobal, row, offset, n)
        ReduceToFirstBlock("max", tileReducedValueLocal, xyLocal, n)
        tileReducedValueLocal[:8] = maximum(
            tileReducedValueLocal[:8], mLocal[:8])     # mNew
        if offset > 0:                               # 首 tile 没有旧状态，不缩放
            alphaLocal[:8] = exp(mLocal[:8] - tileReducedValueLocal[:8])
        xyLocal[:n] = exp(xyLocal[:n] - Broadcast(tileReducedValueLocal[:8]))
        mLocal[:8] = tileReducedValueLocal[:8]         # 保存 mNew，再复用结果区
        ReduceToFirstBlock("sum", tileReducedValueLocal, xyLocal, n)
        if offset == 0:
            sLocal[:8] = tileReducedValueLocal[:8]     # 首 tile 直接初始化分母
        else:
            FusedMulAdd(sLocal, alphaLocal, tileReducedValueLocal, 8)
            # sLocal = sLocal * alphaLocal + tileReducedValueLocal
        # 不写 GM；向量读完后按队列规则归还 xy 缓冲
    sLocal[:8] = onesLocal[:8] / sLocal[:8]            # 此后保存 1/s


def Stage2(row):
    for offset in range(0, cols, tileCols):
        n = min(tileCols, cols - offset)
        xyLocal = CopyIn(xGlobal, row, offset, n)      # 重新读 x
        xyLocal[:n] = exp(xyLocal[:n] - Broadcast(mLocal[:8]))
        xyLocal[:n] *= Broadcast(sLocal[:8])
        CopyOut(yGlobal, row, offset, xyLocal, n)      # 直接写最终结果
```

阶段公式与 UB 状态的对应关系：

```text
Stage1 每 tile：
  mNew    = max(mOld, max(xTile))                 -> tileReducedValueLocal_ 首 32B
  alpha   = exp(mOld-mNew)，首 tile 不计算          -> alphaLocal_
  tileSum = sum(exp(xTile-mNew))                  -> 复用 tileReducedValueLocal_
  sNew    = tileSum                      首 tile   -> sLocal_
          = sOld*alpha + tileSum         后续 tile -> sLocal_，用 FusedMulAdd
  m       = mNew                                  -> mLocal_
Stage1 行末：invS = 1/s                           -> 覆盖 sLocal_
Stage2：      y[j] = exp(x[j]-m)*invS
```

表里的 m、s 是公式符号，真实数据始终在 UB，没有 `GetValue`/`SetValue`。注意顺序：新最大值先写进 `mLocal_`，归约结果区才覆盖为 tileSum。

## 耗时与取舍

真实 Ascend910B3、40 个向量核、CANN 9.1.1、`-O3`，每个 shape 运行 5 次取平均，只计 kernel。口径见[性能记录](../../README.md#性能对照)。

| shape | v04 / us | v05 / us | v06 / us |
|---|---:|---:|---:|
| `(1024,1024)` | 19.276 | 19.052 | 19.500 |
| `(1024,16384)` | 56.432 | 55.728 | 55.608 |
| `(128,131072)` | 116.664 | 112.984 | 93.040 |

两条整行路径没动，不足 1us 的差异是波动。长行比 v04 快约 20.3%，比 v05 快约 17.7%：减少搬运的收益超过了多算一遍指数的成本。非对齐长行 `(128,131071)` 的独立对照里 v06（101.587us）同样低于 v04（130.120us）和 v05（134.067us）。这些结果只说明所测 shape 上的收益，不能推广到所有行长。

## 运行与阅读代码

在仓库根目录加载 CANN 环境并完成 CMake 配置后：

```bash
cmake --build build --target softmax_membase_v06 -j2
./build/Samples/2_Performance/softmax_membase_story/softmax_membase_v06 --core-num 40 --shape 128 131072
```

从 [GENERAL 的两遍流程](softmax_general.h) 开始读，再看 [容量计算与启动](softmax.asc) 和 [归约辅助函数](softmax_common.h)。完整构建、精度标准和采集命令见[执行教程](../../run-guide.md)。
