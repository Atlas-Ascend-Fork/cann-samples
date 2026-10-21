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

// 完整行 DB：ping/pong 各恰好一行，x/y 各两个物理 buffer。
// 高级Reduce RA折半 + 单repeat WholeReduce/Brcb，归约结果全程留在Vector。
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
        const int64_t coreGmOffset = firstRow * data_.cols;
        xGlobal_.SetGlobalBuffer(xGmAddr + coreGmOffset);
        yGlobal_.SetGlobalBuffer(yGmAddr + coreGmOffset);

        // 初始化TQue和TBuf，整行搬入后在UB内完成计算。
        pipe_->InitBuffer(xQue_, 2, data_.xyLocalBytes);
        pipe_->InitBuffer(yQue_, 2, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(mBuf_, UB_REPEAT_SIZE);
        pipe_->InitBuffer(sBuf_, UB_REPEAT_SIZE);

        // 准备TBuf对应的LocalTensor，它们不参与TQue同步。
        tmpLocal_ = tmpBuf_.Get<float>();
        mLocal_ = mBuf_.Get<float>();
        sLocal_ = sBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        for (int64_t r = 0; r < rowsCurrCore_; ++r) {
            const int64_t rowGmOffset = r * data_.cols;
            // 1. 拷入当前整行。
            xRowLocal_ = xQue_.AllocTensor<float>();
            CopyIn(xRowLocal_, xGlobal_[rowGmOffset]);
            xQue_.EnQue(xRowLocal_);

            // 2. 数据就绪后，在UB内完成整行Softmax。
            xRowLocal_ = xQue_.DeQue<float>();
            yRowLocal_ = yQue_.AllocTensor<float>();
            Compute();
            xQue_.FreeTensor(xRowLocal_);

            // 3. 写回结果，再释放输出buffer。
            yQue_.EnQue(yRowLocal_);
            yRowLocal_ = yQue_.DeQue<float>();
            CopyOut(yGlobal_[rowGmOffset], yRowLocal_);
            yQue_.FreeTensor(yRowLocal_);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> yQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_, mBuf_, sBuf_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xRowLocal_, yRowLocal_, tmpLocal_, mLocal_, sLocal_;

    // 整行数据驻留UB，依次计算最大值、指数分子和归一化结果。
    __aicore__ inline void Compute()
    {
        // 1. 求整行最大值并广播，结果留在Vector。
        ReduceToFirstBlock<true>(mLocal_, xRowLocal_, tmpLocal_, data_.cols);
        // 2. 计算exp(x-max)，Sub用高维切分的Counter模式来复用mLocal_的首32B（8个元素都一样）。
        // 高维切分+Counter模式的repeatTimes参数无效，固定填1即可
        // repeatParams={dstBlkStride=1, src0BlkStride=1, src1BlkStride=0表示一直在首个block内打转,
        // dstRepStride=8, src0RepStride=8, src1RepStride=0表示一直在首个repeat内打转}
        uint64_t counterMask[2] = {data_.cols, 0};
        AscendC::SetMaskCount();
        AscendC::Sub<float, true>(yRowLocal_, xRowLocal_, mLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
        AscendC::SetMaskNorm(); // 恢复Normal，供后续Exp/归约使用。
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(yRowLocal_, yRowLocal_, data_.cols);
        AscendC::PipeBarrier<PIPE_V>();
        // 3. 求分母并计算倒数，sLocal_首32B保存相同值。
        ReduceToFirstBlock<false>(sLocal_, yRowLocal_, tmpLocal_, data_.cols);
        AscendC::Duplicate(sLocal_[UB_BLOCK_ELEMS], 1.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(sLocal_, sLocal_[UB_BLOCK_ELEMS], sLocal_, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // 4. 高维Counter段：Mul以零步长复用首32B的分母倒数。
        AscendC::SetMaskCount();
        AscendC::Mul<float, true>(yRowLocal_, yRowLocal_, sLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
        AscendC::SetMaskNorm();
        // m/s/tmp 为跨行单份TBuf；前一行的向量计算结束后再复用。
        AscendC::PipeBarrier<PIPE_V>();
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
};
