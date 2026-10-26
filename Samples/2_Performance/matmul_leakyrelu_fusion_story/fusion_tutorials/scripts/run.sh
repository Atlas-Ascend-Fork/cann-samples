# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
    local dir="$SCRIPT_DIR"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/.ci/build.sh" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}
REPO_ROOT="$(find_repo_root || true)"
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: cannot locate repo root containing .ci/build.sh"
    exit 1
fi
INSTALL_DIR="${REPO_ROOT}/build_out/2_Performance/matmul_leakyrelu_fusion_story/fusion_tutorials"
DEFAULT_TUTORIAL_TARGET="step01_matmul_single_core_vector"

TARGET=""
SKIP_BUILD=false
M=""
K=""
N=""

usage() {
    cat <<'EOF'
Usage: bash run.sh [OPTIONS] [m k n]

One-stop script: build, run, and verify (precision + speed) a fusion tutorial step.
Test data generation and golden check are embedded in each .asc (1024 random
points vs double-precision CPU golden); the program prints [PASS] and ms/iter.

Positional arguments:
  m k n   Matrix dims (m: rows of A; k: cols of A / rows of B; n: cols of B),
          multiples of 16, with k and n <= 65535 (Nd2NzParams.srcDValue is
          uint16_t on dav-2201). Only accepted by Step 9/9s
          (general-shape versions); all other steps are fixed at 4096x4096x4096.

Options:
  --target <name>    Tutorial executable to run.
                      When omitted, defaults to Step 1: step01_matmul_single_core_vector.
  --skip-build       Skip the build/install phase (reuse existing build_out).
  -h, --help         Show this help message and exit.

Available tutorial executables:
  step01_matmul_single_core_vector            (Step 1: 单核向量初始版; 注意运行约需数十秒)
  step02_matmul_cube_fused_mix                (Step 2: 高级 API mix 融合, 反例)
  step03_matmul_fused_2d                      (Step 3: 融合二维切分)
  step04_matmul_decoupled_mix                 (Step 4: 解耦融合 SyncAll)
  step05_matmul_snake_split                   (Step 5: 蛇形窗口调度拆分)
  step06_matmul_lowlevel_split                (Step 6: 低阶 API 拆分)
  step06b_matmul_lowlevel_split_4buf_aswt     (Step 6b: step06 + L1 四级缓冲 + ASWT)
  step07_matmul_leakyrelu_fused_lowlevel      (Step 7: 低阶 mix 逐 tile 流水融合)
  step08_matmul_leakyrelu_fused_lowlevel_aswt (Step 8: step07 + L1 四级缓冲 + ASWT, 当前最优)
  step09_matmul_leakyrelu_fused_general       (Step 9: step08 任意 shape 泛化, m k n)
  step09s_matmul_lowlevel_split_general       (Step 9s: step09 的拆分对照, m k n)

Examples:
  # 构建并运行默认 Step 1（4096^3，注意需数十秒）
  bash scripts/run.sh

  # 指定 Step 运行
  bash scripts/run.sh --target step05_matmul_snake_split

  # 泛化版指定 shape（m k n）
  bash scripts/run.sh --target step09_matmul_leakyrelu_fused_general 4160 4080 4048

  # 跳过构建阶段
  bash scripts/run.sh --target step08_matmul_leakyrelu_fused_lowlevel_aswt --skip-build

EOF
}

KNOWN_TARGETS=(
    step01_matmul_single_core_vector
    step02_matmul_cube_fused_mix
    step03_matmul_fused_2d
    step04_matmul_decoupled_mix
    step05_matmul_snake_split
    step06_matmul_lowlevel_split
    step06b_matmul_lowlevel_split_4buf_aswt
    step07_matmul_leakyrelu_fused_lowlevel
    step08_matmul_leakyrelu_fused_lowlevel_aswt
    step09_matmul_leakyrelu_fused_general
    step09s_matmul_lowlevel_split_general
)

is_known_target() {
    local t
    for t in "${KNOWN_TARGETS[@]}"; do
        [[ "$1" == "$t" ]] && return 0
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ -z "${2:-}" ]] && { echo "ERROR: --target requires a value"; usage; exit 1; }
            TARGET="$2"; shift 2 ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "ERROR: unknown option: $1"; usage; exit 1 ;;
        *)
            if   [[ -z "$M" ]]; then M="$1"
            elif [[ -z "$K" ]]; then K="$1"
            elif [[ -z "$N" ]]; then N="$1"
            else echo "ERROR: unexpected argument: $1"; usage; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    TARGET="${DEFAULT_TUTORIAL_TARGET}"
    echo "    Using default Step 1 target: $TARGET"
fi
if ! is_known_target "$TARGET"; then
    echo "ERROR: unknown target: $TARGET"
    usage
    exit 1
fi
if [[ -n "$M" && "$TARGET" != step09* ]]; then
    echo "ERROR: m k n 仅 Step 9/9s（泛化版）接受；$TARGET 固定 4096x4096x4096"
    exit 1
fi

# ── 1. Build ────────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" != true ]]; then
    echo "=== Building project (via .ci/build.sh dav-2201) ==="
    bash "${REPO_ROOT}/.ci/build.sh" dav-2201
fi

if [[ ! -x "${INSTALL_DIR}/${TARGET}" ]]; then
    echo "ERROR: Executable not found or not executable: ${INSTALL_DIR}/${TARGET}"
    echo "       Run without --skip-build to perform a full build first."
    exit 1
fi

# ── 2. Run ──────────────────────────────────────────────────────────────────
cd "$INSTALL_DIR"
echo "=== Running ${TARGET} ${M} ${K} ${N} ==="
LOG="$(mktemp)"
set +e
if [[ -n "$M" ]]; then
    "./${TARGET}" "$M" "$K" "$N" 2>&1 | tee "$LOG"
else
    "./${TARGET}" 2>&1 | tee "$LOG"
fi
RC=${PIPESTATUS[0]}
set -e

# ── 3. Verify: precision ([PASS]) + speed (ms/iter) ──────────────────────────
echo "=== Verify ==="
if [[ $RC -ne 0 ]]; then
    echo "[FAIL] 程序异常退出 (rc=$RC)"
    rm -f "$LOG"
    exit 1
fi
if grep -q '\[PASS\]' "$LOG"; then
    echo "精度校验: [PASS]（1024 随机点 vs double 精度 CPU golden）"
else
    echo "精度校验: [FAIL]（未检测到 [PASS]）"
    rm -f "$LOG"
    exit 1
fi
grep -E 'ms/iter' "$LOG" | tail -1 | sed 's/^/耗时: /'
rm -f "$LOG"
