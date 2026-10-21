# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""有限输入的FP64 stable Softmax Golden；按ops-test-kit mixed FP32标准校验。"""

from pathlib import Path
import sys
import numpy as np

# https://gitcode.com/cann/ops-test-kit/blob/master/docs/Precision_Comparison.md
RTOL = 2**-10
ATOL = 2**-16
REQUIRED_MATCHED_RATIO = 0.99
MAX_ABS_ERROR_FLOOR = 1e-2

directory = Path(sys.argv[1])
rows, cols = map(int, sys.argv[2:4])
x = (
    np.fromfile(directory / "input.bin", dtype=np.float32)
    .reshape(rows, cols)
    .astype(np.float64)
)
y = np.fromfile(directory / "output.bin", dtype=np.float32).reshape(rows, cols)
if not np.isfinite(x).all() or not np.isfinite(y).all():
    print(
        "[Verify]     mode=mixed, error=non-finite input/output\n[Result]     FAIL",
        flush=True,
    )
    sys.exit(1)
expected = np.exp(x - x.max(axis=-1, keepdims=True))
expected /= expected.sum(axis=-1, keepdims=True)
actual = y.astype(np.float64)
golden = expected  # 保留 FP64 Golden，仅在计算 ULP 锚点时收窄到 FP32。
error = np.abs(actual - golden)
max_abs_error = float(error.max())
# ULP 锚点是最大有限误差位置的 Golden，收窄到实际输出 dtype FP32。
max_error_index = int(np.argmax(error))
g_low = np.float32(golden.flat[max_error_index])
ulp = float(np.spacing(np.abs(g_low)))
max_abs_error_limit = max(MAX_ABS_ERROR_FLOOR, 32 * ulp)
matched = (error <= ATOL + RTOL * np.abs(golden)) & (error <= max_abs_error_limit)
matched_count = np.count_nonzero(matched)
matched_ratio = matched_count / y.size
passed = bool(
    matched_ratio >= REQUIRED_MATCHED_RATIO and max_abs_error <= max_abs_error_limit
)

# 统计相对误差及行和误差只用于诊断，不参与 mixed 的 PASS 判定。
relative = error / (np.abs(golden) + 1e-7)
mere = float(relative.mean())
mare = float(relative.max())
sum_error = np.abs(y.sum(axis=-1, dtype=np.float64) - 1).max(initial=0.0)
print(
    f"[Verify]     mode=mixed, rtol={RTOL:.6e}, atol={ATOL:.6e}\n"
    f"[Verify]     matched={matched_count}/{y.size}({matched_ratio * 100:.6g}%), "
    f"required={REQUIRED_MATCHED_RATIO * 100:.6g}%\n"
    f"[Verify]     maxAbsError={max_abs_error:.6g}, limit={max_abs_error_limit:.6e}\n"
    f"[Diag]       MERE={mere:.6g}, MARE={mare:.6g}\n"
    f"[Diag]       maxSumError={sum_error:.6g}\n"
    f"[Result]     {'PASS' if passed else 'FAIL'}",
    flush=True,
)
sys.exit(0 if passed else 1)
