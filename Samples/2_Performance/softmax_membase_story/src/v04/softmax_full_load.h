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
// 基础Reduce仍保留V_S/GetValue；归约工作空间由tmpLocal_管理。
class SoftmaxKernelOpFullLoad {
public:
    __aicore__ inline SoftmaxKernelOpFullLoad(AscendC::TPipe* pipe, const SoftmaxTilingData& data)
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
        const int64_t coreGmOffset = firstRow * data_.cols;
        xGlobal_.SetGlobalBuffer(xGmAddr + coreGmOffset);
        yGlobal_.SetGlobalBuffer(yGmAddr + coreGmOffset);

        // 初始化TQue和TBuf，整行搬入后在UB内完成计算。
        pipe_->InitBuffer(xyQue_, 1, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(reducedValueBuf_, UB_BLOCK_SIZE);

        // 准备TBuf对应的LocalTensor，它们不参与TQue同步。
        tmpLocal_ = tmpBuf_.Get<float>();
        reducedValueLocal_ = reducedValueBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        for (int64_t r = 0; r < rowsCurrCore_; ++r) {
            const int64_t rowGmOffset = r * data_.cols;
            // 1. 拷入整行；同一buffer随后原地计算并搬出。
            xyRowLocal_ = xyQue_.AllocTensor<float>();
            CopyIn(xyRowLocal_, xGlobal_[rowGmOffset]);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyRowLocal_);

            // 2. 数据就绪后，原地计算整行Softmax。
            xyRowLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();
            Compute();
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyRowLocal_);

            // 3. 写回结果，搬出完成后才能复用同一buffer。
            xyRowLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            CopyOut(yGlobal_[rowGmOffset], xyRowLocal_);
            xyQue_.FreeTensor(xyRowLocal_);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::TPipe* pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> xyQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_, reducedValueBuf_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xyRowLocal_, tmpLocal_, reducedValueLocal_;

    // 整行数据驻留UB，依次计算最大值、指数分子和归一化结果。
    __aicore__ inline void Compute()
    {
        // 1. 求整行最大值；GetValue前等待V->S。
        AscendC::ReduceMax(reducedValueLocal_, xyRowLocal_, tmpLocal_, data_.cols, false);
        SetWaitFlag<AscendC::HardEvent::V_S>();
        const float maximum = reducedValueLocal_.GetValue(0);

        // 2. 减去最大值，计算指数分子。
        AscendC::Adds(xyRowLocal_, xyRowLocal_, -maximum, data_.cols);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(xyRowLocal_, xyRowLocal_, data_.cols);
        AscendC::PipeBarrier<PIPE_V>();

        // 3. 求分母并归一化；reducedValueLocal_复用同一32B，工作空间仍为tmpLocal_。
        // 前n个数据接口仅处理有效列，无需补0到完整repeat。
        AscendC::ReduceSum(reducedValueLocal_, xyRowLocal_, tmpLocal_, data_.cols);
        SetWaitFlag<AscendC::HardEvent::V_S>();
        const float invDenom = 1.0f / reducedValueLocal_.GetValue(0);
        AscendC::Muls(xyRowLocal_, xyRowLocal_, invDenom, data_.cols);
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
        const auto id = pipe_->FetchEventID(event);
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }
};
