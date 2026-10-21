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

        // 初始化TQue和TBuf：xy开DB，tmp和在线统计空间各一份。
        pipe_->InitBuffer(xyQue_, 2, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(rowStatsBuf_, 4 * UB_BLOCK_SIZE);

        // 准备TBuf对应的LocalTensor，它们不参与TQue同步。
        tmpLocal_ = tmpBuf_.Get<float>();
        reduceTmpLocal_ = tmpLocal_;
        reduceTmpLocal_.SetSize(data_.reduceTmpBytes / sizeof(float));
        // tmp前区供高级RA使用；尾部256B先存mNew，再复用为tileSum。
        tileReducedValueLocal_ = tmpLocal_[data_.reduceTmpBytes / sizeof(float)];
        auto rowStatsLocal = rowStatsBuf_.Get<float>();
        mLocal_ = rowStatsLocal;
        sLocal_ = rowStatsLocal[UB_BLOCK_ELEMS];
        alphaLocal_ = rowStatsLocal[2 * UB_BLOCK_ELEMS];
        onesLocal_ = rowStatsLocal[3 * UB_BLOCK_ELEMS];
        AscendC::Duplicate(onesLocal_, 1.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        // 默认Normal；高维Counter段在各阶段单独标注。
        // 遍历本核分配的行。
        for (int64_t r = 0; r < rowsCurrCore_; r++) {
            const int64_t rowGmOffset = r * data_.cols;
            // 第一轮在线更新最大值和分母，不向GM暂存分子。
            Stage1WithOnlineStats(rowGmOffset);
            // 第二轮重读x，计算并写回最终结果。
            Stage2(rowGmOffset);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    AscendC::TPipe* pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> xyQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_, rowStatsBuf_;
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    AscendC::LocalTensor<float> xyTileLocal_, tmpLocal_, reduceTmpLocal_, tileReducedValueLocal_;
    AscendC::LocalTensor<float> mLocal_, sLocal_, alphaLocal_, onesLocal_;

    // 单遍在线更新整行最大值和分母，最终将分母倒数存入sLocal_。
    __aicore__ inline void Stage1WithOnlineStats(int64_t rowGmOffset)
    {
        AscendC::Duplicate(mLocal_, -__builtin_inff(), UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            // 尾tile只处理剩余有效列。
            const uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            xyTileLocal_ = xyQue_.AllocTensor<float>();
            CopyInGm2UbFP32<true, 1>(xyTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyTileLocal_);

            // 2. 更新mNew，计算当前tileSum
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();

            // mNew=Max(ReduceMax(x), mOld)
            ReduceToFirstBlock<true>(tileReducedValueLocal_, xyTileLocal_, reduceTmpLocal_, currTileCols);
            AscendC::Max(tileReducedValueLocal_, tileReducedValueLocal_, mLocal_, UB_BLOCK_ELEMS);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileGmOffset > 0) {
                // alpha=Exp(mOld-mNew)，将已有分母换到新最大值基准。
                AscendC::Sub(alphaLocal_, mLocal_, tileReducedValueLocal_, UB_BLOCK_ELEMS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(alphaLocal_, alphaLocal_, UB_BLOCK_ELEMS);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // tileSum=ReduceSum(Exp(x-mNew))
            // 高维切分Counter段：src1以零block/repeat步长复用mNew首32B。
            uint64_t counterMask[2] = {currTileCols, 0};
            AscendC::SetMaskCount();
            AscendC::Sub<float, true>(xyTileLocal_, xyTileLocal_, tileReducedValueLocal_, counterMask, 1,
                                      {1, 1, 0, 8, 8, 0});
            AscendC::SetMaskNorm(); // 恢复Normal，供后续Exp/归约使用。
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(xyTileLocal_, xyTileLocal_, currTileCols);
            // 保存mNew后，tileReducedValueLocal_可复用为tileSum；与上面的Exp无数据依赖。
            AscendC::Adds(mLocal_, tileReducedValueLocal_, 0.0f, UB_BLOCK_ELEMS);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceToFirstBlock<false>(tileReducedValueLocal_, xyTileLocal_, reduceTmpLocal_, currTileCols);

            // 3. 合并tileSum：sNew=sOld*alpha+tileSum。
            if (tileGmOffset == 0) {
                // 有限输入保证首tile有有效最大值；首轮不计算旧状态的缩放。
                AscendC::Adds(sLocal_, tileReducedValueLocal_, 0.0f, UB_BLOCK_ELEMS);
            } else {
                // FusedMulAdd以dst旧值作为乘数。
                AscendC::FusedMulAdd(sLocal_, alphaLocal_, tileReducedValueLocal_, UB_BLOCK_ELEMS);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // stage1不写GM，但仍需V->MTE3交接，防止Free后MTE2提前覆盖输入。
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyTileLocal_);
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            xyQue_.FreeTensor(xyTileLocal_);
        }
        // 行末将分母s转换为倒数(1/s)，供第二轮直接相乘。
        AscendC::Div(sLocal_, onesLocal_, sLocal_, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 重读输入，逐元素计算exp(x-max)乘以1/s并写回结果。
    __aicore__ inline void Stage2(int64_t rowGmOffset)
    {
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            const uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            xyTileLocal_ = xyQue_.AllocTensor<float>();
            CopyInGm2UbFP32<false>(xyTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xyQue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(xyTileLocal_);

            // 2. 根据整行最大值和分母倒数计算输出。
            // y=exp(x-m)*(1/s)
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, float>();
            uint64_t counterMask[2] = {currTileCols, 0};
            // 高维切分Counter段：整段反复读取行最大值首32B。
            AscendC::SetMaskCount();
            AscendC::Sub<float, true>(xyTileLocal_, xyTileLocal_, mLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
            AscendC::SetMaskNorm();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(xyTileLocal_, xyTileLocal_, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();
            // 高维切分Counter段：整段反复读取行分母倒数首32B。
            AscendC::SetMaskCount();
            AscendC::Mul<float, true>(xyTileLocal_, xyTileLocal_, sLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
            AscendC::SetMaskNorm();
            xyQue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM>(xyTileLocal_);

            // 3. 写回当前tile的最终结果。
            xyTileLocal_ = xyQue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, float>();
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], xyTileLocal_, currTileCols);
            xyQue_.FreeTensor(xyTileLocal_);
        }
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
};
