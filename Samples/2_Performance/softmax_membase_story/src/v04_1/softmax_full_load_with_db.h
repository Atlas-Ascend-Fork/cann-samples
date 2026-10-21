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

// 完整行DB：xy共用ping/pong两份buffer，ms按max→sum→倒数顺序复用。
class SoftmaxKernelOpFullLoadWithDB {
public:
    __aicore__ inline SoftmaxKernelOpFullLoadWithDB(const SoftmaxTilingData& data)
        : data_{data}
    {}

    __aicore__ inline void Init(__gm__ float* xGmAddr, __gm__ float* yGmAddr)
    {
        // 按完整行分核，初始化本核GlobalTensor的地址。
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        rowsCurrCore_ = data_.baseRowsPerCore + (blockIdx < data_.extraRows ? 1 : 0);
        const int64_t firstRow = static_cast<int64_t>(blockIdx) * data_.baseRowsPerCore +
                                 (blockIdx < data_.extraRows ? blockIdx : data_.extraRows);
        xGlobal_.SetGlobalBuffer(xGmAddr + firstRow * data_.cols);
        yGlobal_.SetGlobalBuffer(yGmAddr + firstRow * data_.cols);

        // 显式分配xy双buffer、单份tmp和统计区。
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> allocator;
        xyStorageLocal_ = allocator.Alloc<float>(2 * data_.xyLocalBytes / sizeof(float));
        tmpLocal_ = allocator.Alloc<float>(data_.tmpLocalBytes / sizeof(float));
        reducedValueLocal_ = allocator.Alloc<float>((UB_REPEAT_SIZE + UB_BLOCK_SIZE) / sizeof(float));
        // 前256B供归约使用，末32B的常量1用于计算分母倒数。
        onesLocal_ = reducedValueLocal_[UB_REPEAT_ELEMS];
        AscendC::Duplicate(onesLocal_, 1.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // 反向token：首次两个slot都可写，随后由CopyOut完成的MTE3归还。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    }

    __aicore__ inline void Process()
    {
        uint32_t slot = 0;
        for (int64_t row = 0; row < rowsCurrCore_; ++row) {
            const int64_t offset = row * data_.cols;
            auto xyRowLocal = xyStorageLocal_[slot * (data_.xyLocalBytes / sizeof(float))];
            // 同一份xy既接收输入又供CopyOut读，必须等MTE3读完才能覆盖。
            // 1. 等待当前slot空闲，再拷入整行。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            CopyIn(xyRowLocal, xGlobal_[offset]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

            // 2. MTE2->V等待后，在UB内完成整行计算。
            Compute(xyRowLocal);

            // 3. 等待Vector完成，写回结果并归还slot。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
            CopyOut(yGlobal_[offset], xyRowLocal);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            slot ^= 1; // 下一行切换ping/pong。
        }
        // 消费最后归还的token，也覆盖只有一行时从未使用的pong token。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xyStorageLocal_, tmpLocal_, reducedValueLocal_, onesLocal_;

    // 整行数据驻留UB，依次计算最大值、指数分子和归一化结果。
    __aicore__ inline void Compute(const AscendC::LocalTensor<float>& xyRowLocal)
    {
        const uint32_t cols = data_.cols;
        // 1. 求整行最大值并广播，结果留在Vector。
        ReduceToFirstBlock<true>(reducedValueLocal_, xyRowLocal, tmpLocal_, cols);

        // 2. 原地计算exp(x-max)，Sub用高维切分的Counter模式来复用reducedValueLocal_的首32B（8个元素都一样）。
        // 高维切分+Counter模式的repeatTimes参数无效，固定填1即可
        // repeatParams={dstBlkStride=1, src0BlkStride=1, src1BlkStride=0表示一直在首个block内打转,
        // dstRepStride=8, src0RepStride=8, src1RepStride=0表示一直在首个repeat内打转}
        uint64_t counterMask[2] = {cols, 0};
        AscendC::SetMaskCount();
        AscendC::Sub<float, true>(xyRowLocal, xyRowLocal, reducedValueLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
        AscendC::SetMaskNorm();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(xyRowLocal, xyRowLocal, cols);
        AscendC::PipeBarrier<PIPE_V>();

        // 3. reducedValueLocal_由最大值复用为分母及其倒数，归约仍使用tmpLocal_。
        ReduceToFirstBlock<false>(reducedValueLocal_, xyRowLocal, tmpLocal_, cols);
        AscendC::Div(reducedValueLocal_, onesLocal_, reducedValueLocal_, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // 4. 高维Counter段：Mul以零步长复用首32B的分母倒数。
        AscendC::SetMaskCount();
        AscendC::Mul<float, true>(xyRowLocal, xyRowLocal, reducedValueLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
        AscendC::SetMaskNorm();
        AscendC::PipeBarrier<PIPE_V>(); // 下一行才可覆盖单份ms/tmp。
    }

    // 拷入整行，不满32B的尾块补-Inf。
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float>& dstLocal,
                                  const AscendC::GlobalTensor<float>& srcGlobal)
    {
        AscendC::DataCopyExtParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = data_.cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams{};
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (UB_BLOCK_ELEMS - data_.cols % UB_BLOCK_ELEMS) % UB_BLOCK_ELEMS;
        padParams.paddingValue = -__builtin_inff();
        AscendC::DataCopyPad<float>(dstLocal, srcGlobal, copyParams, padParams);
    }

    // 只写回本行有效列，避免覆盖下一行。
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float>& dstGlobal,
                                   const AscendC::LocalTensor<float>& srcLocal)
    {
        AscendC::DataCopyExtParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = data_.cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad<float>(dstGlobal, srcLocal, copyParams);
    }
};
