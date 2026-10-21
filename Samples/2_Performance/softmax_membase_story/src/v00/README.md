# v00：单核三遍计算

[返回 Softmax 总览](../../README.md) · 下一版：[v01 按完整行分核](../v01/README.md)

v00 是基线版本，只用一个向量核。它要解决两个基本问题：一行太长、装不进片上工作区（UB）时怎么切开算；列数不是对齐整数倍时，尾块怎么处理。后续所有版本都沿用这套切分和尾部规则。

输入、输出都是 FP32，支持任意正整数列数，不需要额外的片外临时空间。输入要求是有限数值，不处理 NaN、Inf。

## 三遍算一行

SafeSoftmax 对一行 `x` 的计算：

```text
m    = max(x[0], ..., x[cols-1])
p[j] = exp(x[j] - m)
s    = sum(p[0], ..., p[cols-1])
y[j] = p[j] / s
```

先减最大值再取指数，指数的输入最大为 0，不会溢出。例如 `x=[0,0]`：m=0，分子是 `[1,1]`，s=2，输出 `[0.5,0.5]`。

难点在于 m 和 s 都是整行的统计量，要看完整行才能算出来，而一行可能远超 UB 容量。v00 的做法是把一行算三遍：

![三遍计算与片外中间结果](../../images/alg/softmax_three_pass.png)

1. **第一遍**：读 x，分段求最大值，合并出整行最大值 m。
2. **第二遍**：再读 x，算分子 `p=exp(x-m)` 和分母 s，把 p 写进输出 y 的 GM 空间暂存。
3. **第三遍**：从 y 读回 p，乘 `1/s` 写回，得到最终结果。

分子直接借用输出空间暂存，不用额外申请 GM 工作区。第三遍开始前要等第二遍写出完成，保证读回的是分子。

“归约”指把一组数合成一个结果，这里指求最大值和求和。每段的归约结果先落在 UB，等向量计算完成后用 `GetValue` 取成标量，再合并到整行结果里。

即使整行装得下 UB，v00 也走三遍。按每行 `cols` 个元素计：读 x 两次、读 y 一次、写 y 两次，共请求搬运 `20*cols` 字节。这个数用来比较各版本的搬运量，不等于实际片外流量。

## 长行切成 tile

一行装不进 UB 时，沿列切成若干段，每段叫一个 tile。`tileCols` 是一段最多包含的有效元素数，由 host 按 UB 容量反推：

- UB 里要放 x、y 缓冲各一份，归约临时区一份，最大值和分母各 32B；
- x、y 每份容量按 `tileCols` 向上对齐到 256B（一个 repeat），方便求和前填充尾部；
- 扣除这些固定开销后，剩余空间能放多少个元素，就是 `tileCols`。

在 Ascend910B3 上可用 UB 为 196352B，算出 `tileCols=24064`。`cols=131072` 的行就切成 5 个 24064 列的整段，加一段 10752 列的尾段。计算过程在 [softmax.asc](softmax.asc) 的 `ComputeSoftmaxTiling` 里，kernel 不写死这个值。

每个 tile 的实际有效长度是：

```text
currTileCols = min(tileCols, cols - 当前tile偏移)
```

整段用 `tileCols`，最后一段用剩余列数。三个阶段都按这个长度循环，不会把缓冲容量当成有效数据。

## 非对齐列数的尾块处理

列数可以不是 8 或 64 的整数倍。搬运和归约对尾部的要求不同，要分开处理：

- **搬运**：`DataCopyPad` 按实际字节数读写。不满 32B 的尾块可以补齐搬入；写回只写有效元素，不会覆盖下一行。
- **求最大值**：搬入时把不满 32B 的尾块补 `-Inf`，`ReduceMax` 只归约有效元素数。`-Inf` 不影响最大值，它只是内部填充值，和“输入不含 Inf”的约定不冲突。
- **求分母**：先只对有效元素算指数，再用按位 mask 的 `Duplicate` 把最后一个 repeat 的无效位置补 0，然后对完整 repeat 做 `ReduceSum`。0 不影响总和。

以 65 个元素为例：第一个 repeat 装 64 个有效分子；第二个 repeat 只有 1 个有效分子，其余 63 个位置补 0 后再求和。

![65 个 FP32 的有效区间与归约补齐](../../images/alg/softmax_tail_padding.png)

图中按元素编号标出有效区与补齐区，宽度不代表元素数量比例。[编辑图源](../../images/alg/softmax_tail_padding.excalidraw)。

`Duplicate` 从最后一个 repeat 的起点开始，按位选中无效位置，不覆盖前面的有效分子。注意：缓冲申请了 256B 对齐的容量，不代表尾部已经填好，填充必须显式做。

## 单核与单列

| tilingKey | 条件 | 做法 |
|---|---|---|
| `SINGLE_TAIL` | `cols=1` | 不读输入，输出全部填 1.0。 |
| `GENERAL` | `cols>1` | 一个核依次算完所有行，每行三遍。 |

v00 固定启动一个核，不接受 `--core-num`。

## 三个目标 shape

| shape | 分段方式 | 观察点 |
|---|---|---|
| `(1024,1024)` | 整行一个 tile，仍走三遍 | 短行也在重复搬运 |
| `(1024,16384)` | 整行一个 tile，仍走三遍 | 整行驻留能省掉片外往返 |
| `(128,131072)` | `5×24064 + 10752` | 长行要合并 6 段的最大值和分母 |

Ascend910B3、CANN 9.1.1，固定单核，kernel 三次平均耗时：

| shape | kernel 耗时（us） |
|---|---:|
| `(1024,1024)` | 1177.493 |
| `(1024,16384)` | 5078.040 |
| `(128,131072)` | 4551.353 |

计时口径见[总览的性能记录](../../README.md#性能对照)。v00 这组数据单独采集，与 v01 起的五次平均不是同一组，不能直接算加速比。

## 代码结构与执行流程

| 文件 | 主要职责 |
|---|---|
| [softmax.asc](softmax.asc) | `ComputeSoftmaxTiling` 计算分块参数，`softmax_k` 选择模板；外围 host 代码负责分配、启动和校验。 |
| [softmax_common.h](softmax_common.h) | 常量、`SoftmaxTilingKey`、`SoftmaxTilingData` 和日志打印。 |
| [softmax_general.h](softmax_general.h) | `Init` 绑定地址、分配 UB；`Process` 逐行调用三个阶段。 |
| [softmax_single_tail.h](softmax_single_tail.h) | 单列输出直接填 1。 |

下面用 Python 风格伪代码保留入口和阶段函数名，简化参数类型、搬运和控制细节。`read_x`、`read_y`、`write_y` 表示数据搬运，不是源码中的函数名。伪代码省略了队列操作和阶段内部的流水同步，实际代码里这些同步不能少。

```python
# softmax_k：host 按 cols 选择模板实例
if cols == 1:
    op = SoftmaxKernelOpSingleTail()
    op.Init(yGmAddr, data)
    op.Process()                         # 本核输出全部填为 1.0
else:
    op = SoftmaxKernelOpGeneral(pipe, data)
    op.Init(xGmAddr, yGmAddr)
    op.Process()

# SoftmaxKernelOpGeneral.Process
for r in range(data.rows):
    rowGmOffset = r * cols
    maximum = Stage1WithMaximum(rowGmOffset)
    sum = Stage2WithDenomSum(rowGmOffset, maximum)
    wait_for_numerator_writeback()       # 第 2 遍写 y 完成，才允许第 3 遍读 y
    Stage3(rowGmOffset, 1.0 / sum)

# Stage1WithMaximum：m = max(各 tile 的最大值)
def Stage1WithMaximum(rowGmOffset):
    maximum = -inf
    for offset in range(0, cols, tileCols):
        n = min(tileCols, cols - offset)
        xTileLocal_ = read_x(rowGmOffset + offset, n)
        currMaximum = ComputeTileMaximum(n)  # ReduceMax -> 等待 V_S -> GetValue
        maximum = max(maximum, currMaximum)
    return maximum

# Stage2WithDenomSum：p = exp(x-m)，s += sum(p)
def Stage2WithDenomSum(rowGmOffset, maximum):
    sum = 0.0
    for offset in range(0, cols, tileCols):
        n = min(tileCols, cols - offset)
        xTileLocal_ = read_x(rowGmOffset + offset, n)
        yTileLocal_ = exp(xTileLocal_[:n] - maximum)
        sum += ComputeTileExpSum(n)       # 尾 repeat 补 0，再 ReduceSum/GetValue
        write_y(rowGmOffset + offset, yTileLocal_[:n])
    return sum

# Stage3：y = p/s；分子 p 已由上一阶段暂存在输出 GM 中
def Stage3(rowGmOffset, invDenom):
    for offset in range(0, cols, tileCols):
        n = min(tileCols, cols - offset)
        xTileLocal_ = read_y(rowGmOffset + offset, n)
        yTileLocal_ = xTileLocal_[:n] * invDenom
        write_y(rowGmOffset + offset, yTileLocal_[:n])
```

数学量 `m`、`s` 在代码中分别是标量 `maximum`、`sum`；`mLocal_`、`sLocal_` 保存当前 tile 的归约结果，经 `GetValue` 取出后才合并到整行标量。`yTileLocal_` 在第二遍保存分子，第三遍保存最终输出。

## 构建与运行

加载 CANN 环境后，在仓库根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-2201
cmake --build build --target softmax_membase_v00 -j2
./build/Samples/2_Performance/softmax_membase_story/softmax_membase_v00 --shape 7 65
```

构建、校验和性能采集的完整用法见[执行教程](../../run-guide.md)。
