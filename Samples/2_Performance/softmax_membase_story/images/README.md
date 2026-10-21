# Softmax 文档配图

每张图保留同名 `.excalidraw` 源文件和导出 `.png`。修改时先编辑图源，再重新导出；README 使用 PNG，图注提供源文件链接。

所有图均为算法、数据关系或缓冲依赖示意，不是仿真或实卡时间轴。图形使用规整边框和无衬线字体，区域大小以标签说明为准。

| 图源 | 内容 |
| --- | --- |
| [三阶段计算](alg/softmax_three_pass.excalidraw) | 求最大值、求分子与分母、归一化，以及中间分子的 GM 去向 |
| [完整行分核](alg/softmax_row_partition.excalidraw) | 连续行按商和余数分配给各核 |
| [模板选择](alg/softmax_tiling_paths.excalidraw) | 单列、整行双缓冲、整行单缓冲和分块计算 |
| [尾块补齐](alg/softmax_tail_padding.excalidraw) | 32B 搬运边界、256B 归约边界与有效写回范围 |
| [归约与广播](alg/softmax_ra_reduce.excalidraw) | RA、WholeReduce、Brcb 与首 32B 数据块复用 |
| [UB 复用](memory/softmax_ub_reuse.excalidraw) | x/y 合并和整行统计量的生命周期 |
| [GENERAL 缓冲](memory/softmax_general_buffers.excalidraw) | 临时工作区、tile 归约结果与行状态 |
| [手动同步](alg/softmax_manual_sync.excalidraw) | 同槽的正向交接、反向复用及初始化和排空 |
| [逐位置累计](alg/softmax_accumulate.excalidraw) | tile 间累计、短尾保留与行末归约 |
| [在线算法](alg/softmax_online.excalidraw) | 最大值与分母递推，以及两遍输入扫描 |

可在 Excalidraw 中打开源文件，编辑后导出 PNG。使用 `draw-ascendc-pipeline-diagram` 技能的批量导出工具时，先安装 Python Playwright、对应 Chromium 和中文字体；本例使用 Noto Sans CJK SC。Playwright 版本记录在本目录的 `requirements.txt` 中。

若该技能安装在 `$HOME/.codex/skills`，可从仓库根目录执行：

```bash
pip3 install -r Samples/2_Performance/softmax_membase_story/images/requirements.txt
python3 -m playwright install chromium

skill_dir="$HOME/.codex/skills/draw-ascendc-pipeline-diagram"
for source in Samples/2_Performance/softmax_membase_story/images/{alg,memory}/*.excalidraw; do
    python3 "$skill_dir/scripts/validate_excalidraw.py" "$source"
    python3 "$skill_dir/scripts/render_excalidraw.py" "$source"
done
```

导出工具会加载 Excalidraw 的前端模块，需要网络访问。导出后还应检查中文缺字、文本遮挡、连线端点和 Markdown 显示宽度下的可读性。
