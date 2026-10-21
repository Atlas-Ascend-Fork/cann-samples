# 构建、运行与仿真

以下命令从 cann-samples 仓库根目录执行。先按本机安装位置加载 CANN Toolkit 的 `set_env.sh`，确认 `ASCEND_HOME_PATH` 指向所需工具链。本样例已验证 CANN 9.1.1 与 Ascend910B3；设备信息可用 `asys info -r=status` 查询。

## 编译与运行

```bash
pip3 install -r Samples/2_Performance/softmax_membase_story/requirements.txt
cmake -S . -B build -DNPU_ARCH=dav-2201
cmake --build build --target softmax_membase_story -j4

./build/Samples/2_Performance/softmax_membase_story/softmax_membase_v00 --shape 7 65
./build/Samples/2_Performance/softmax_membase_story/softmax_membase_v06 --core-num 3 --shape 7 65
```

聚合目标构建 v00～v06 和 v04_1，共八个实现。单独构建用 `--target softmax_membase_v06`。可执行文件和公共的 `softmax_gen.py`、`softmax_verify.py` 都在 `build/Samples/2_Performance/softmax_membase_story/` 下。更换工具链或目标架构时使用不同的构建目录。

v00 固定单核，只接受 `--shape rows cols`；其余版本额外接受 `--core-num N` 作为核数上限，默认用平台 AIV 核数。实际核数和 TilingKey 会在日志中打印。不传 shape 时默认 `(160,131072)`。

程序自动完成造数、H2D、kernel 启动、D2H 和 FP64 参考值校验。`[Result] PASS` 且退出码为 0 表示通过；mixed 精度规则见[总说明](README.md#编译运行与校验)。数据保存在可执行文件旁的 `data/softmax_membase_vXX/`，同一版本的多个进程不要共用这个目录。

八个版本各运行三个目标 shape（各一次，失败即停止）：

```bash
Samples/2_Performance/softmax_membase_story/scripts/quicksmoke_softmax.sh
```

三个脚本都按自身路径定位仓库，可以在任意目录执行：

```bash
/path/to/cann-samples/Samples/2_Performance/softmax_membase_story/scripts/rebuild_softmax.sh
/path/to/cann-samples/Samples/2_Performance/softmax_membase_story/scripts/quicksmoke_softmax.sh
/path/to/cann-samples/Samples/2_Performance/softmax_membase_story/scripts/sim-rebuild_softmax.sh
```

rebuild 用 Release、`SIM=OFF` 配置仓库的 `build/`，只构建 `softmax_membase_story`，保留已有构建目录。用 `JOBS=8 /path/to/rebuild_softmax.sh` 调整并行度，默认 4。

quicksmoke 先检查默认目录中的八个可执行文件和两份 Python 脚本：缺了就调用同目录的 rebuild，齐全就直接运行，不检查源码时间戳。修改源码后请主动 rebuild。也可以传入一个已构建的可执行目录，此时只运行该目录中的程序。

quicksmoke 用于功能对照；它的总运行时间包含 Python 造数和校验，不是 kernel 耗时。

## 安装目录

按仓库的统一安装流程构建并安装后，文件位于 `build_out/2_Performance/softmax_membase_story/`：

```bash
cmake --build build -j4
cmake --install build --prefix build_out
./build_out/2_Performance/softmax_membase_story/softmax_membase_v06 --core-num 3 --shape 7 65
```

统一安装会包含当前架构下的全部样例，所以上面的构建命令没有指定单个 target。本专题的可执行文件、两份 Python 脚本和 quicksmoke 脚本会一同安装，移动时保持在同一目录。只运行本样例时无需安装步骤。

## 实卡 kernel 耗时

性能采集使用 Release 配置，保持 `-O3` 且不带 `-g`，与总文档的测量口径一致。以 v06 的长行为例：

```bash
cmake -S . -B build -DNPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target softmax_membase_story -j4

msopprof \
  --application="$PWD/build/Samples/2_Performance/softmax_membase_story/softmax_membase_v06 --core-num 40 --shape 128 131072" \
  --output="$PWD/build/profiling/softmax_membase_v06" \
  --aic-metrics=BasicInfo --warm-up=0 --launch-count=1 --replay-mode=kernel
```

每次采集使用新的输出目录。从 `OpBasicInfo.csv` 读取 kernel 的 `Task Duration(us)`，并核对实际核数及核频率。对比时固定硬件、CANN、shape、dtype 和编译选项，交替采集各版本并保留原值与平均值。

[性能对照](README.md#性能对照)只计 kernel，不计造数、校验、ACL 初始化及 H2D/D2H。主线表使用每版每 shape 五次平均，v00 和 v04_1 的独立三次平均另外列示。

## msopprof simulator

sim-rebuild 使用独立的 `build-sim/`，对应以下配置：

```bash
cmake -S . -B build-sim -DNPU_ARCH=dav-2201 -DSIM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-sim --target softmax_membase_story -j4
```

`SIM` 在根 `CMakeLists.txt` 中定义，仅用于 `dav-2201`。开启后，在 ASC 初始化前设置 `CMAKE_ASC_ARCHITECTURES=dav-2201` 和 `CMAKE_ASC_RUN_MODE=sim`，由工具链统一配置模型库及搜索路径。Softmax 子目录只负责为仿真目标添加 `-g`，同时保留 `-O3`。实卡脚本显式传入 `SIM=OFF`，两种产物分目录保存。

在已加载 CANN 9.1.1 环境的终端中，从仓库根目录采集一个单核小用例：

```bash
msopprof simulator --soc-version=Ascend910B3 --dump=on \
  --aic-metrics=PipeUtilization \
  --output="$PWD/build-sim/profiling/softmax_membase_v06" \
  "$PWD/build-sim/Samples/2_Performance/softmax_membase_story/softmax_membase_v06" \
  --core-num 1 --shape 7 65
```

每次采集使用新的输出目录。实测流水在报告的 `OPPROF_*/simulator/core0.veccore0/trace.json` 中；先看应用校验 PASS、工具提示 `All task success` 且解析完成。观察相邻行的双缓冲可把 shape 改为 `32 1024`，长行可用 `1 131072`。仿真资源消耗较大，从小 shape 开始。
