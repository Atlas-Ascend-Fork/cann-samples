# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""仅生成有限FP32输入：固定种子、uniform[-4,4]，不注入Inf/NaN且计算不溢出。"""

from pathlib import Path
import sys
import numpy as np

directory = Path(sys.argv[1])
rows, cols = map(int, sys.argv[2:4])
directory.mkdir(parents=True, exist_ok=True)
x = np.random.default_rng(20261008).uniform(-4, 4, (rows, cols)).astype(np.float32)
x.tofile(directory / "input.bin")
