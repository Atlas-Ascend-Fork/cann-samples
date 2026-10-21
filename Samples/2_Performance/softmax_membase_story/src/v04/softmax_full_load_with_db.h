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
    __aicore__ inline SoftmaxKernelOpFullLoadWithDB(AscendC::TPipe* pipe, const SoftmaxTilingData& data)
        : data_{data},
          pipe_{pipe}
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

        // 初始化TQue和TBuf，整行搬入后在UB内完成计算。
        pipe_->InitBuffer(xyQue_, 2, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(reducedValueAndOnesBuf_, UB_REPEAT_SIZE + UB_BLOCK_SIZE);

        // 准备TBuf对应的LocalTensor，它们不参与TQue同步。
        tmpLocal_ = tmpBuf_.Get<float>();
        reducedValueLocal_ = reducedValueAndOnesBuf_.Get<float>();
        // 前256B供归约使用，末32B的常量1用于计算分母倒数。
        onesLocal_ = reducedValueLocal_[UB_REPEAT_ELEMS];
        AscendC::Duplicate(onesLocal_, 1.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = 0; row < rowsCurrCore_; ++row) {
            const int64_t offset = row * data_.cols;
            // 1. 拷入当前整行，ping/pong交替使用。
            auto xyRowLocal = xyQue_.AllocTensor<float>();
            CopyIn(xyRowLocal, xGlobal_[offset]);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyRowLocal);

            // 2. 数据就绪后，原地计算整行Softmax。
            xyRowLocal = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();
            Compute(xyRowLocal);
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyRowLocal);

            // 3. 写回结果，搬出完成后才能复用同一buffer。
            xyRowLocal = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            CopyOut(yGlobal_[offset], xyRowLocal);
            xyQue_.FreeTensor(xyRowLocal);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::TPipe* pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> xyQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_, reducedValueAndOnesBuf_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> tmpLocal_, reducedValueLocal_, onesLocal_;

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
