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

// 整行无DB：输入原地变为输出，max/sum顺序复用一个32B统计区。
// WholeReduce分级归约后仍通过V_S/GetValue读取Scalar；归约工作空间由tmpLocal_管理。
class SoftmaxKernelOpFullLoad {
public:
    __aicore__ inline SoftmaxKernelOpFullLoad(const SoftmaxTilingData& data)
        : data_{data}
    {}

    __aicore__ inline void Init(__gm__ float* xGmAddr, __gm__ float* yGmAddr)
    {
        // 按完整行分核，初始化本核GlobalTensor的地址。
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        rowsCurrCore_ = data_.baseRowsPerCore + (blockIdx < data_.extraRows ? 1 : 0);
        const int64_t firstRow = static_cast<int64_t>(blockIdx) * data_.baseRowsPerCore +
                                 (blockIdx < data_.extraRows ? blockIdx : data_.extraRows);
        const int64_t coreGmOffset = firstRow * data_.cols;
        xGlobal_.SetGlobalBuffer(xGmAddr + coreGmOffset);
        yGlobal_.SetGlobalBuffer(yGmAddr + coreGmOffset);

        // InitSocState由kernel入口统一调用；2201的静态UB分配从0开始。
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> allocator;
        xyRowLocal_ = allocator.Alloc<float>(data_.xyLocalBytes / sizeof(float));
        tmpLocal_ = allocator.Alloc<float>(data_.tmpLocalBytes / sizeof(float));
        reducedValueLocal_ = allocator.Alloc<float, UB_BLOCK_ELEMS>();
    }

    __aicore__ inline void Process()
    {
        // 单buffer：MTE3确认上次搬出后，MTE2才能覆盖xy；首轮预置一个空闲token。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        for (int64_t r = 0; r < rowsCurrCore_; ++r) {
            const int64_t rowGmOffset = r * data_.cols;
            // 1. 等待当前slot空闲，再拷入整行。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            CopyIn(xyRowLocal_, xGlobal_[rowGmOffset]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            // 2. MTE2->V等待后，在UB内完成整行计算。
            Compute();

            // 3. 等待Vector完成，写回结果并归还slot。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            CopyOut(yGlobal_[rowGmOffset], xyRowLocal_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        // 消费最后一次发布，既排空末行搬出，也不遗留未消费事件。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xyRowLocal_, tmpLocal_, reducedValueLocal_;

    // 整行数据驻留UB，依次计算最大值、指数分子和归一化结果。
    __aicore__ inline void Compute()
    {
        // 1. 求整行最大值；GetValue前等待V->S。
        ReduceScalar<true>();
        SetWaitFlag<AscendC::HardEvent::V_S>();
        const float maximum = reducedValueLocal_.GetValue(0);

        // 2. 减去最大值，计算指数分子。
        AscendC::Adds(xyRowLocal_, xyRowLocal_, -maximum, data_.cols);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(xyRowLocal_, xyRowLocal_, data_.cols);
        AscendC::PipeBarrier<PIPE_V>();

        // 3. 求分母并归一化，reducedValueLocal_的32B空间与max复用。
        // 分级WholeReduce前将尾repeat补0，避免无效lane计入分母。
        ReduceScalar<false>();
        SetWaitFlag<AscendC::HardEvent::V_S>();
        const float invDenom = 1.0f / reducedValueLocal_.GetValue(0);
        AscendC::Muls(xyRowLocal_, xyRowLocal_, invDenom, data_.cols);
    }

    // 基础Reduce依赖TPipe事件池，静态模式改用公开WholeReduce分级归约。
    // 全程Normal：每个repeat归约64lane；最后一层只归约有效partial数量。
    template <bool isMax>
    __aicore__ inline void ReduceScalar()
    {
        PadReductionTail<isMax>(xyRowLocal_, data_.cols);
        const uint32_t repeats = (data_.cols + UB_REPEAT_ELEMS - 1) / UB_REPEAT_ELEMS;
        // 248既不超过uint8 repeat上限，又是8的倍数，使下一段dst保持32B对齐。
        constexpr uint32_t chunkRepeats = 248;
        for (uint32_t offset = 0; offset < repeats; offset += chunkRepeats) {
            const uint32_t currRepeats = repeats - offset < chunkRepeats ? repeats - offset : chunkRepeats;
            if constexpr (isMax) {
                AscendC::WholeReduceMax<float, true>(tmpLocal_[offset], xyRowLocal_[offset * UB_REPEAT_ELEMS],
                                                     UB_REPEAT_ELEMS, currRepeats, 1, 1, 8,
                                                     AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            } else {
                AscendC::WholeReduceSum<float, true>(tmpLocal_[offset], xyRowLocal_[offset * UB_REPEAT_ELEMS],
                                                     UB_REPEAT_ELEMS, currRepeats, 1, 1, 8);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto partialLocal = tmpLocal_;
        uint32_t partialCount = repeats;
        if (repeats > UB_REPEAT_ELEMS) {
            PadReductionTail<isMax>(tmpLocal_, repeats);
            partialCount = (repeats + UB_REPEAT_ELEMS - 1) / UB_REPEAT_ELEMS;
            // tmp源前缀已补齐到partialCount*64，第二层写其后，源/目的不重叠。
            partialLocal = tmpLocal_[partialCount * UB_REPEAT_ELEMS];
            if constexpr (isMax) {
                AscendC::WholeReduceMax<float, true>(partialLocal, tmpLocal_, UB_REPEAT_ELEMS, partialCount, 1, 1, 8,
                                                     AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            } else {
                AscendC::WholeReduceSum<float, true>(partialLocal, tmpLocal_, UB_REPEAT_ELEMS, partialCount, 1, 1, 8);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (isMax) {
            AscendC::WholeReduceMax<float, true>(reducedValueLocal_, partialLocal, partialCount, 1, 1, 1, 8,
                                                 AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            AscendC::WholeReduceSum<float, true>(reducedValueLocal_, partialLocal, partialCount, 1, 1, 1, 8);
        }
    }

    // 拷入整行，不满32B的尾块补-Inf。
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float>& dstLocal,
                                  const AscendC::GlobalTensor<float>& srcGlobal)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = data_.cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
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
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = data_.cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad<float>(dstGlobal, srcLocal, copyParams);
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void SetWaitFlag()
    {
        AscendC::SetFlag<event>(0);
        AscendC::WaitFlag<event>(0);
    }
};
