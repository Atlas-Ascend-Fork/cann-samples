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

# 各版本各跑三个典型 shape；缺少默认构建产物时先调用同目录的构建脚本。
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: $0 [executable_dir]"
    echo "Default: <repo>/build/Samples/2_Performance/softmax_membase_story; build automatically if needed."
    echo "After install, default to the script directory. A custom executable_dir must already be built."
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [executable_dir]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$script_dir/../CMakeLists.txt" ]]; then
    repo_root=$(cd -- "$script_dir/../../../.." && pwd)
    default_dir="$repo_root/build/Samples/2_Performance/softmax_membase_story"
else
    # 安装包中的脚本与可执行文件同目录。
    default_dir="$script_dir"
fi
executable_dir=${1:-"$default_dir"}
versions=(v00 v01 v02 v03 v04 v04_1 v05 v06)
shapes=("1024 1024" "1024 16384" "128 131072")

artifacts_ready()
{
    local version
    for version in "${versions[@]}"; do
        [[ -x "$executable_dir/softmax_membase_$version" ]] || return 1
    done
    [[ -f "$executable_dir/softmax_gen.py" && -f "$executable_dir/softmax_verify.py" ]]
}

if ! artifacts_ready; then
    if [[ "$executable_dir" == "$default_dir" && -x "$script_dir/rebuild_softmax.sh" ]]; then
        echo "[Smoke]      Build artifacts missing; running rebuild_softmax.sh"
        "$script_dir/rebuild_softmax.sh"
    fi
    if ! artifacts_ready; then
        echo "[ERROR]      Missing executables or Python scripts in: $executable_dir" >&2
        exit 1
    fi
fi

passed=0
for version in "${versions[@]}"; do
    for shape in "${shapes[@]}"; do
        read -r rows cols <<< "$shape"
        echo "[Smoke]      version=$version, shape=($rows,$cols)"
        if ! "$executable_dir/softmax_membase_$version" --shape "$rows" "$cols"; then
            echo "[Smoke]      FAIL, version=$version, shape=($rows,$cols)" >&2
            exit 1
        fi
        passed=$((passed + 1))
        echo
    done
done
echo "[Smoke]      PASS, cases=$passed"
