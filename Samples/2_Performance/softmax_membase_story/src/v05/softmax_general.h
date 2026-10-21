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

class SoftmaxKernelOpGeneral {
public:
    __aicore__ inline SoftmaxKernelOpGeneral(AscendC::TPipe* pipe, const SoftmaxTilingData& data)
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

        // xy仍开DB；acc按位置累计各tile，Stage1/2之间复用同一份空间。
        pipe_->InitBuffer(xyQue_, 2, data_.xyLocalBytes);
        pipe_->InitBuffer(accBuf_, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(rowStatsBuf_, 3 * UB_BLOCK_SIZE);

        // 准备TBuf对应的LocalTensor，它们不参与TQue同步。
        tmpLocal_ = tmpBuf_.Get<float>();
        reduceTmpLocal_ = tmpLocal_;
        reduceTmpLocal_.SetSize(data_.reduceTmpBytes / sizeof(float));
        // tmp前区供RA工作，后区专存256B的当前tile归约结果。
        tileReducedValueLocal_ = tmpLocal_[data_.reduceTmpBytes / sizeof(float)];
        accLocal_ = accBuf_.Get<float>();
        auto rowStatsLocal = rowStatsBuf_.Get<float>();
        mLocal_ = rowStatsLocal;
        sLocal_ = rowStatsLocal[UB_BLOCK_ELEMS];
        onesLocal_ = rowStatsLocal[2 * UB_BLOCK_ELEMS];
        AscendC::Duplicate(onesLocal_, 1.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        // 默认Normal；高维Counter段在各阶段单独标注。
        // 遍历本核分配的行。
        for (int64_t r = 0; r < rowsCurrCore_; r++) {
            const int64_t rowGmOffset = r * data_.cols; // 本核内的行偏移。
            // 第一轮计算整行最大值。
            Stage1WithMaximum(rowGmOffset);
            // 第二轮计算分子和分母，将分子暂存在yGlobal_。
            Stage2WithDenomSum(rowGmOffset);
            // 等待分子写回GM完成，再由第三轮读回。
            SetWaitFlag<AscendC::HardEvent::MTE3_MTE2>();
            // 第三轮读回分子，乘以分母倒数。
            Stage3(rowGmOffset);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::TPipe* pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> xyQue_;
    // m/s同时存活，占用独立32B块；归约临时空间统一在tmpLocal_。
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_, rowStatsBuf_, accBuf_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xyTileLocal_, tmpLocal_, reduceTmpLocal_, mLocal_, sLocal_, onesLocal_,
        tileReducedValueLocal_, accLocal_;

    // 逐位置合并各tile的最大值，行末统一归约到mLocal_。
    __aicore__ inline void Stage1WithMaximum(int64_t rowGmOffset)
    {
        const uint32_t accCols = data_.xyLocalBytes / sizeof(float);
        AscendC::Duplicate(accLocal_, -__builtin_inff(), accCols);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            // 尾tile只处理剩余有效列。
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            xyTileLocal_ = xyQue_.AllocTensor<float>();
            CopyInGm2UbFP32<true, 1>(xyTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyTileLocal_);

            // 2. 按位置合并最大值，延后到行末归约。
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();
            // 只更新有效列，acc其余位置保留前面tile的累计值，无需补齐尾repeat。
            AscendC::Max(accLocal_, accLocal_, xyTileLocal_, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();

            // 不写GM仍需V->MTE3交接，防止Free后MTE2提前覆盖输入。
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyTileLocal_);
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            xyQue_.FreeTensor(xyTileLocal_);
        }
        // 高位置保留前面tile的最大值，因此行末归约完整acc。
        ReduceToFirstBlock<true>(tileReducedValueLocal_, accLocal_, reduceTmpLocal_, accCols);
        AscendC::Adds(mLocal_, tileReducedValueLocal_, 0.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 将分子暂存到yGlobal_并逐位置累加，行末归约后将分母倒数存入sLocal_。
    __aicore__ inline void Stage2WithDenomSum(int64_t rowGmOffset)
    {
        const uint32_t accCols = data_.xyLocalBytes / sizeof(float);
        AscendC::Duplicate(accLocal_, 0.0f, accCols);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            xyTileLocal_ = xyQue_.AllocTensor<float>();
            CopyInGm2UbFP32<false>(xyTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyTileLocal_);

            // 2. 计算分子及分母贡献。
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();

            // 高维Counter段：mask[0]为tile列数，src1零步长复用mLocal_首32B。
            uint64_t counterMask[2] = {currTileCols, 0};
            AscendC::SetMaskCount();
            AscendC::Sub<float, true>(xyTileLocal_, xyTileLocal_, mLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
            AscendC::SetMaskNorm(); // 恢复Normal，供后续Exp/归约使用。
            AscendC::PipeBarrier<PIPE_V>();

            // 原地计算有效元素的指数，只累加有效列，无需将尾repeat补0。
            AscendC::Exp(xyTileLocal_, xyTileLocal_, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accLocal_, accLocal_, xyTileLocal_, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();

            // 3. 将分子暂存到yGlobal_，供第三轮读回。
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyTileLocal_);
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], xyTileLocal_, currTileCols);
            xyQue_.FreeTensor(xyTileLocal_);
        }
        // 只在行末归约一次sum，结果仍通过Vector广播，不读回Scalar。
        ReduceToFirstBlock<false>(tileReducedValueLocal_, accLocal_, reduceTmpLocal_, accCols);
        AscendC::Div(sLocal_, onesLocal_, tileReducedValueLocal_, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 从yGlobal_读回分子，乘以分母倒数并原地写回。
    __aicore__ inline void Stage3(int64_t rowGmOffset)
    {
        // 高维Counter段覆盖整个循环；Mul内部更新每个tile的计数。
        AscendC::SetMaskCount();
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 读回暂存在yGlobal_的分子。
            xyTileLocal_ = xyQue_.AllocTensor<float>();
            CopyInGm2UbFP32<false>(xyTileLocal_, yGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyTileLocal_);

            // 2. 乘以sLocal_中的分母倒数。
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();
            uint64_t counterMask[2] = {currTileCols, 0};
            AscendC::Mul<float, true>(xyTileLocal_, xyTileLocal_, sLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});

            // 3. 写回当前tile的最终结果。
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyTileLocal_);
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], xyTileLocal_, currTileCols);
            xyQue_.FreeTensor(xyTileLocal_);
        }
        // Counter段结束，恢复Normal；后续Vector接口自行设置mask。
        AscendC::SetMaskNorm();
    }

    // GM拷入UB：isPad时尾块补齐32B，padMode=0补0，否则补-Inf。
    // isPad=false时只使用有效元素，不依赖尾块填充值。
    template <bool isPad = false, int padMode = 0>
    __aicore__ inline void CopyInGm2UbFP32(const AscendC::LocalTensor<float>& dstLocal,
                                           const AscendC::GlobalTensor<float>& srcGlobal, uint32_t elemNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elemNum * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = isPad;
        padParams.leftPadding = 0;
        if constexpr (isPad) {
            padParams.rightPadding = (UB_BLOCK_ELEMS - elemNum % UB_BLOCK_ELEMS) % UB_BLOCK_ELEMS;
            if constexpr (padMode == 0) {
                padParams.paddingValue = 0.0f;
            } else {
                // 对不满32B的尾块pad补-inf方便ReduceMax
                padParams.paddingValue = -__builtin_inff();
            }
        } else {
            padParams.rightPadding = 0;
            padParams.paddingValue = 0.0f;
        }
        AscendC::DataCopyPad<float>(dstLocal, srcGlobal, copyParams, padParams);
    }

    // 仅拷出有效元素，避免覆盖下一行。
    __aicore__ inline void CopyOutUb2GmFP32(const AscendC::GlobalTensor<float>& dstGlobal,
                                            const AscendC::LocalTensor<float>& srcLocal, uint32_t elemNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elemNum * sizeof(float);
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

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        T1 tb = static_cast<T1>(b);
        return static_cast<T1>(a + tb - 1) / tb;
    }

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return CeilDiv<T1, T2>(a, b) * static_cast<T1>(b);
    }
};
