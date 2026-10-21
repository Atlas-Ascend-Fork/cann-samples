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
    __aicore__ inline SoftmaxKernelOpGeneral(const SoftmaxTilingData& data)
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

        // 分配xy双buffer、单份tmp及m/s/ones；buffer复用由Flag保护。
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> allocator;
        const uint32_t xyLocalCols = data_.xyLocalBytes / sizeof(float);
        xyLocal_[0] = allocator.Alloc<float>(xyLocalCols);
        xyLocal_[1] = allocator.Alloc<float>(xyLocalCols);
        tmpLocal_ = allocator.Alloc<float>(data_.tmpLocalBytes / sizeof(float));
        reduceTmpLocal_ = tmpLocal_;
        reduceTmpLocal_.SetSize(data_.reduceTmpBytes / sizeof(float));
        // tmp前区供RA工作，后区专存256B的当前tile归约结果。
        tileReducedValueLocal_ = tmpLocal_[data_.reduceTmpBytes / sizeof(float)];
        auto rowStatsLocal = allocator.Alloc<float, 3 * UB_BLOCK_ELEMS>();
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
            // Stage2末已等待分子写回GM，第三轮可安全读回。
            // 第三轮读回分子，乘以分母倒数。
            Stage3(rowGmOffset);
        }
    }

private:
    const SoftmaxTilingData data_;
    uint32_t rowsCurrCore_;
    // 两个slot分别使用事件id 0/1；输入输出共用对应的xyLocal_。
    AscendC::GlobalTensor<float> xGlobal_, yGlobal_;
    // m/s同时存活，占用独立32B块；归约临时空间统一在tmpLocal_。
    AscendC::LocalTensor<float> xyLocal_[2], tmpLocal_, reduceTmpLocal_, mLocal_, sLocal_, onesLocal_,
        tileReducedValueLocal_;

    // 逐tile归约最大值，将整行最大值广播保存在mLocal_中。
    __aicore__ inline void Stage1WithMaximum(int64_t rowGmOffset)
    {
        AscendC::Duplicate(mLocal_, -__builtin_inff(), UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // 本阶段没有GM输出：Vector发布V_MTE2，允许MTE2重新填充对应slot。
        InitReverseTokens<AscendC::HardEvent::V_MTE2>();
        uint32_t slot = 0;
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            // 尾tile只处理剩余有效列。
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            auto xyTileLocal = xyLocal_[slot];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
            CopyInGm2UbFP32<true, 1>(xyTileLocal, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot); // MTE2发布输入就绪。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

            // 2. 归约当前tile并更新行最大值。
            ReduceToFirstBlock<true>(tileReducedValueLocal_, xyTileLocal, reduceTmpLocal_, currTileCols);
            AscendC::Max(mLocal_, mLocal_, tileReducedValueLocal_, UB_BLOCK_ELEMS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot); // Vector读完，归还slot。
            slot ^= 1;                                          // 下一轮切换ping/pong。
        }
        DrainReverseTokens<AscendC::HardEvent::V_MTE2>();
    }

    // 将exp(x-max)暂存到yGlobal_，累加分母并将其倒数存入sLocal_。
    __aicore__ inline void Stage2WithDenomSum(int64_t rowGmOffset)
    {
        AscendC::Duplicate(sLocal_, 0.0f, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // 有GM输出：MTE3发布MTE3_MTE2，允许MTE2覆盖已经搬出的slot。
        InitReverseTokens<AscendC::HardEvent::MTE3_MTE2>();
        uint32_t slot = 0;
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 拷入当前tile。
            auto xyTileLocal = xyLocal_[slot];
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            CopyInGm2UbFP32<false>(xyTileLocal, xGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot); // MTE2发布输入就绪。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

            // 2. 计算分子及分母贡献。
            // 高维Counter段：mask[0]为tile列数，src1零步长复用mLocal_首32B。
            uint64_t counterMask[2] = {currTileCols, 0};
            AscendC::SetMaskCount();
            AscendC::Sub<float, true>(xyTileLocal, xyTileLocal, mLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});
            AscendC::SetMaskNorm(); // 恢复Normal，供后续Exp/归约使用。
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(xyTileLocal, xyTileLocal, currTileCols);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceToFirstBlock<false>(tileReducedValueLocal_, xyTileLocal, reduceTmpLocal_, currTileCols);
            AscendC::Add(sLocal_, sLocal_, tileReducedValueLocal_, UB_BLOCK_ELEMS);
            AscendC::PipeBarrier<PIPE_V>();

            // 3. 将分子暂存到yGlobal_，供第三轮读回。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot); // Vector发布分子就绪。
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], xyTileLocal, currTileCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot); // MTE3搬完，归还slot。
            slot ^= 1;                                             // 下一轮切换ping/pong。
        }
        // 在MTE2泳道等待两个slot写回完成，下一阶段才能读回分子。
        DrainReverseTokens<AscendC::HardEvent::MTE3_MTE2>();
        // 在Vector上计算分母倒数，首32B保持相同值以供广播。
        AscendC::Div(sLocal_, onesLocal_, sLocal_, UB_BLOCK_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 从yGlobal_读回分子，乘以分母倒数并原地写回。
    __aicore__ inline void Stage3(int64_t rowGmOffset)
    {
        // 高维Counter段覆盖整个循环；Mul内部更新每个tile的计数。
        AscendC::SetMaskCount();
        InitReverseTokens<AscendC::HardEvent::MTE3_MTE2>();
        uint32_t slot = 0;
        for (int64_t tileGmOffset = 0; tileGmOffset < data_.cols; tileGmOffset += data_.tileCols) {
            uint32_t currTileCols =
                (tileGmOffset + data_.tileCols) <= data_.cols ? data_.tileCols : data_.cols - tileGmOffset;
            // 1. 读回暂存在yGlobal_的分子。
            auto xyTileLocal = xyLocal_[slot];
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            CopyInGm2UbFP32<false>(xyTileLocal, yGlobal_[rowGmOffset + tileGmOffset], currTileCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot); // MTE2发布输入就绪。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

            // 2. 乘以sLocal_中的分母倒数。
            uint64_t counterMask[2] = {currTileCols, 0};
            AscendC::Mul<float, true>(xyTileLocal, xyTileLocal, sLocal_, counterMask, 1, {1, 1, 0, 8, 8, 0});

            // 3. 写回当前tile的最终结果。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot); // Vector发布最终结果就绪。
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
            CopyOutUb2GmFP32(yGlobal_[rowGmOffset + tileGmOffset], xyTileLocal, currTileCols);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot); // MTE3搬完，归还slot。
            slot ^= 1;                                             // 下一轮切换ping/pong。
        }
        DrainReverseTokens<AscendC::HardEvent::MTE3_MTE2>();
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

    // 预置两个空闲slot；未使用的slot也由阶段末Wait配对消费。
    template <AscendC::HardEvent event>
    __aicore__ inline void InitReverseTokens()
    {
        AscendC::SetFlag<event>(0);
        AscendC::SetFlag<event>(1);
    }

    // 等待两个slot完成本阶段，并消费剩余token。
    template <AscendC::HardEvent event>
    __aicore__ inline void DrainReverseTokens()
    {
        AscendC::WaitFlag<event>(0);
        AscendC::WaitFlag<event>(1);
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
