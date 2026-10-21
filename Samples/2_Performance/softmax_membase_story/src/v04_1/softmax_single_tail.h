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
        // 单列输出就是一维数组，GlobalTensor直接定位到本核的元素区间。
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const int64_t firstElem = static_cast<int64_t>(data.rows) * blockIdx / blockNum;
        const int64_t endElem = static_cast<int64_t>(data.rows) * (blockIdx + 1) / blockNum;
        yGlobal_.SetGlobalBuffer(yGmAddr + firstElem, endElem - firstElem);
    }

    __aicore__ inline void Process()
    {
        // SDK Fill隐式使用TPipe的stack buffer和事件池，这里显式准备一个repeat。
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> allocator;
        auto onesLocal = allocator.Alloc<float, UB_REPEAT_ELEMS>();
        AscendC::Duplicate(onesLocal, 1.0f, UB_REPEAT_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        // ones全程只读，多个输出块无须反向等待；尾块使用DataCopyPad精确写出。
        const int64_t count = yGlobal_.GetSize();
        for (int64_t offset = 0; offset < count; offset += UB_REPEAT_ELEMS) {
            const uint32_t currElems = count - offset < UB_REPEAT_ELEMS ? count - offset : UB_REPEAT_ELEMS;
            AscendC::DataCopyExtParams params{1, currElems * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad<float>(yGlobal_[offset], onesLocal, params);
        }
    }

private:
    AscendC::GlobalTensor<float> yGlobal_;
};
