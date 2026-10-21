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
        // 当前核需要处理的行数=能整除分给每个核的函数+（余下的行，让前面的核，每个核分一行，直到分完）
        rowsCurrCore_ = data_.baseRowsPerCore + (blockIdx < data_.extraRows ? 1 : 0);
        // 当前核需要处理的第一行位于所有行的位置，若有尾行，则在可以整除行的基础上加上min(当前blockIdx，额外行总数)即可
        const int64_t firstRow = static_cast<int64_t>(blockIdx) * data_.baseRowsPerCore +
                                 (blockIdx < data_.extraRows ? blockIdx : data_.extraRows);
        const int64_t coreGmOffset = firstRow * data_.cols;
        xGlobal_.SetGlobalBuffer(xGmAddr + coreGmOffset);
        yGlobal_.SetGlobalBuffer(yGmAddr + coreGmOffset);

        // 初始化TQue和TBuf
        pipe_->InitBuffer(xQue_, 1, data_.xyLocalBytes);
        pipe_->InitBuffer(yQue_, 1, data_.xyLocalBytes);
        pipe_->InitBuffer(tmpBuf_, data_.tmpLocalBytes);
        pipe_->InitBuffer(mBuf_, UB_BLOCK_SIZE);
        pipe_->InitBuffer(sBuf_, UB_BLOCK_SIZE);

        // 先准备好Buf用的LocalTensor，它们不参与TQue同步
        tmpLocal_ = tmpBuf_.Get<float>();
        mLocal_ = mBuf_.Get<float>();
        sLocal_ = sBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        // 遍历本核分配的行。
        for (int64_t r = 0; r < rowsCurrCore_; r++) {
            const int64_t rowGmOffset = r * data_.cols; // 本核内的行偏移。
            // 第一轮计算整行最大值。
            float maximum = Stage1WithMaximum(rowGmOffset);
            // 第二轮计算分子和分母，将分子暂存在yGlobal_。
            float sum = Stage2WithDenomSum(rowGmOffset, maximum);
            // 第二轮涉及MTE3写回，第三轮又要MTE2读取，因此需要同步：等待分子写回GM完成，再由第三轮读回。
            SetWaitFlag<AscendC::HardEvent::MTE3_MTE2>();
            // 第三轮读回分子，乘以分母倒数。
            Stage3(rowGmOffset, 1.0f / sum);
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
    AscendC::LocalTensor<float> xTileLocal_, yTileLocal_, tmpLocal_, mLocal_, sLocal_;

    // 逐tile归约并合并，返回整行最大值。
    __aicore__ inline float Stage1WithMaximum(int64_t rowGmOffset)
    {
        float maximum = -__builtin_inff();
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            // 当前函数处理的tile行元素的个数，取决于是不是切一行后的尾tile
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile，padMode=1补-inf
            xTileLocal_ = xQue_.AllocTensor<float>();
            CopyInGm2UbFP32<true, 1>(xTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xQue_.EnQue(xTileLocal_);

            // 2. 计算当前tile
            xTileLocal_ = xQue_.DeQue<float>();
            float currMaximum = ComputeTileMaximum(currTileCols);
            xQue_.FreeTensor(xTileLocal_);

            // 3. 和历史tile的结果做合并（更新最大值）
            maximum = maximum > currMaximum ? maximum : currMaximum;
        }
        return maximum;
    }

    // 计算一tile行的最大值
    __aicore__ inline float ComputeTileMaximum(uint32_t elemNum)
    {
        // 使用基础API的连续计算ReduceMax，mask为Counter模式
        AscendC::ReduceMax(mLocal_, xTileLocal_, tmpLocal_, elemNum, false);
        // GetValue属于Scalar流水，需要等待Vector流水计算完，此处为等待ReduceMax
        SetWaitFlag<AscendC::HardEvent::V_S>();
        return mLocal_.GetValue(0);
    }

    // 计算并暂存exp(x-max)也就是分子在输出yGm上，同时计算返回一行的分母和Sum(exp(x-max))。
    __aicore__ inline float Stage2WithDenomSum(int64_t rowGmOffset, float maximum)
    {
        float sum = 0.0f;
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 重新拷入当前tile数据
            xTileLocal_ = xQue_.AllocTensor<float>();
            CopyInGm2UbFP32<false>(xTileLocal_, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xQue_.EnQue(xTileLocal_);

            // 2. 计算当前tile的分子以及累加
            xTileLocal_ = xQue_.DeQue<float>();
            yTileLocal_ = yQue_.AllocTensor<float>();
            AscendC::Adds(yTileLocal_, xTileLocal_, -maximum, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(yTileLocal_, yTileLocal_, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();
            float currTileSum = ComputeTileExpSum(currTileCols);
            sum += currTileSum;
            xQue_.FreeTensor(xTileLocal_);
            yQue_.EnQue(yTileLocal_);

            // 3. 用yGm临时存放算好的分子
            yTileLocal_ = yQue_.DeQue<float>();
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], yTileLocal_, currTileCols);
            yQue_.FreeTensor(yTileLocal_);
        }
        return sum;
    }

    __aicore__ inline float ComputeTileExpSum(uint32_t elemNum)
    {
        // 基础高维切分ReduceSum执行repeat间的二叉树累加。
        // 尾repeat先补0，避免无效lane计入分母。
        int repeatTime = CeilDiv(elemNum, UB_REPEAT_ELEMS);
        uint32_t tailElems = elemNum % UB_REPEAT_ELEMS;
        if (tailElems != 0) {
            // Duplicate起点必须32B对齐：从最后一个256B repeat起点按bit清零无效lane。
            // FP32使用低64位mask（mask[0]）；有效的前tailElems位保持不变。
            uint64_t tailMask[2] = {UINT64_MAX << tailElems, 0};
            AscendC::Duplicate<float>(yTileLocal_[elemNum - tailElems], 0.0f, tailMask, 1, 1,
                                      UB_REPEAT_SIZE / UB_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::ReduceSum(sLocal_, yTileLocal_, tmpLocal_, UB_REPEAT_ELEMS, repeatTime,
                           UB_REPEAT_SIZE / UB_BLOCK_SIZE);
        // GetValue前等待Vector完成归约。
        SetWaitFlag<AscendC::HardEvent::V_S>();
        return sLocal_.GetValue(0);
    }

    // 重读暂存在yGm的每元素分子，乘以分母（denomSum）的倒数并写回结果。
    __aicore__ inline void Stage3(int64_t rowGmOffset, float invDenom)
    {
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 把Stage2写回yGm的分子部分取出来
            xTileLocal_ = xQue_.AllocTensor<float>();
            CopyInGm2UbFP32<false>(xTileLocal_, yGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            xQue_.EnQue(xTileLocal_);

            // 2. 乘以1/分母
            xTileLocal_ = xQue_.DeQue<float>();
            yTileLocal_ = yQue_.AllocTensor<float>();
            AscendC::Muls(yTileLocal_, xTileLocal_, invDenom, currTileCols);
            xQue_.FreeTensor(xTileLocal_);
            yQue_.EnQue(yTileLocal_);

            // 3. 拷回原地
            yTileLocal_ = yQue_.DeQue<float>();
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], yTileLocal_, currTileCols);
            yQue_.FreeTensor(yTileLocal_);
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

    // 把LocalTensor拷出到GM
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
