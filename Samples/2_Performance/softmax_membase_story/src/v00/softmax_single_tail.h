/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <kernel_operator.h>
#include "softmax_common.h"

class SoftmaxKernelOpSingleTail {
public:
    __aicore__ inline void Init(__gm__ float* yGmAddr, const SoftmaxTilingData& data)
    {
        yGlobal_.SetGlobalBuffer(yGmAddr, data.rows);
    }

    __aicore__ inline void Process()
    {
        // 单列softmax直接输出1，不需要读取输入。
        AscendC::Fill(yGlobal_, yGlobal_.GetSize(), 1.0f);
    }

private:
    AscendC::GlobalTensor<float> yGlobal_;
};
