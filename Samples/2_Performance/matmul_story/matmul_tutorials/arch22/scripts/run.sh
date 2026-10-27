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
INSTALL_DIR="${REPO_ROOT}/build_out/2_Performance/matmul_story/matmul_tutorials/arch22"

ALL_TARGETS=(
    step1_matmul_single_core_vector
    step2_matmul_cube_multicore
    step3_matmul_snake_schedule
    step4_matmul_stagger_schedule
    step5_matmul_lowlevel_pipeline
    step6_matmul_lowlevel_4buf_stagger
    step7_matmul_general_shape
    step8_matmul_full_load
)
# 默认跳过 step1（单核基线，4096³ 约 30s+）；需要时 --target step1_matmul_single_core_vector 单独跑
DEFAULT_TARGETS=("${ALL_TARGETS[@]:1}")

SELECTED=()
SKIP_BUILD=false

usage() {
    cat <<'EOF'
Usage: bash run.sh [OPTIONS] [--target <name> ...]

One-stop script: build (dav-2201 only) and run the matmul tutorial steps.
Each step prints embedded golden spot-checks (first-run + post-timing recheck,
both [PASS] required for exit code 0) and one ms/iter line.

Options:
  --target <name>    Executable(s) to run; repeatable. When omitted, runs
                     step2..step8 (skips the slow single-core baseline step1).
  --skip-build       Skip the build/install phase (reuse existing build_out).
  -h, --help         Show this help message and exit.

Available executables (v1 -> v7 optimization ladder + full-load topic):
  step1_matmul_single_core_vector      (v1  single-core vector baseline, ~30s+)
  step2_matmul_cube_multicore          (v2  Cube multi-core, high-level API)
  step3_matmul_snake_schedule          (v3  snake-window scheduling)
  step4_matmul_stagger_schedule        (v4  staggered core partition, isolated)
  step5_matmul_lowlevel_pipeline       (v5  low-level per-tile pipeline)
  step6_matmul_lowlevel_4buf_stagger   (v6  +4-stage L1 + stagger, best)
  step7_matmul_general_shape           (v7  arbitrary-shape, args: <m> <k> <n>)
  step8_matmul_full_load               (full-load template topic, M=32)

Examples:
  bash run.sh                                    # build + run step2..step8
  bash run.sh --target step6_matmul_lowlevel_4buf_stagger
  bash run.sh --skip-build --target step7_matmul_general_shape

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ -z "${2:-}" ]] && { echo "ERROR: --target requires a value"; usage; exit 1; }
            SELECTED+=("$2"); shift 2 ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=("${DEFAULT_TARGETS[@]}")
fi

for t in "${SELECTED[@]}"; do
    if [[ ! " ${ALL_TARGETS[*]} " == *" $t "* ]]; then
        echo "ERROR: unknown target: $t"
        usage
        exit 1
    fi
done

if [[ "$SKIP_BUILD" != true ]]; then
    echo "=== Building (via .ci/build.sh, dav-2201 only) ==="
    bash "${REPO_ROOT}/.ci/build.sh" dav-2201
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "ERROR: Install directory not found: $INSTALL_DIR"
    echo "       Run without --skip-build to perform a full build first."
    exit 1
fi
cd "$INSTALL_DIR"

for t in "${SELECTED[@]}"; do
    echo
    echo "=== Running ${t} ==="
    "./${t}"
done
