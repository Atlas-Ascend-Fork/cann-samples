#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: $0"
    echo "Build all Softmax MemBase versions in <repo>/build-sim; load CANN set_env.sh first."
    echo "Set JOBS to choose build parallelism (default: 4)."
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
build_dir="$repo_root/build-sim"

cmake -S "$repo_root" -B "$build_dir" \
    -DNPU_ARCH=dav-2201 -DSIM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target softmax_membase_story --parallel "${JOBS:-4}"
echo "[Build]      executables=$build_dir/Samples/2_Performance/softmax_membase_story"
