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

#include <cstdint>
#include <cstdio>
#include <kernel_operator.h>
#include <adv_api/reduce/reduce.h>

constexpr int UB_BLOCK_SIZE = 32; // UB 数据块大小，单位为字节。
constexpr int UB_BLOCK_ELEMS = UB_BLOCK_SIZE / static_cast<int>(sizeof(float));
constexpr int UB_REPEAT_SIZE = 256; // 一次 SIMD repeat 处理的字节数。
constexpr int UB_REPEAT_ELEMS = UB_REPEAT_SIZE / static_cast<int>(sizeof(float));

template <typename T1, typename T2>
__host_aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    T1 tb = static_cast<T1>(b);
    return static_cast<T1>(a + tb - 1) / tb;
}

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return CeilDiv<T1, T2>(a, b) * static_cast<T1>(b);
}

enum class SoftmaxTilingKey {
    SINGLE_TAIL,
    FULLLOAD_WITH_DB,
    FULLLOAD,
    GENERAL,
};

struct SoftmaxTilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t baseRowsPerCore; // 每个核基础分配的完整行数。
    uint32_t extraRows;       // 前extraRows个核各多处理一行。
    uint32_t tileCols;
    uint32_t xyLocalBytes;   // 每份xyLocal的容量，单位为字节。
    uint32_t reduceTmpBytes; // 高阶Reduce API使用的临时空间的大小，属于tmpLocalBytes的一部分。
    uint32_t tmpLocalBytes;  // 在GENERAL模板里，除了Reduce API使用的临时空间外，还加了一些别的临时空间。
};

inline void PrintTilingData(const SoftmaxTilingData& data)
{
    std::printf("[TilingData] rows=%u, cols=%u\n"
                "[TilingData] baseRowsPerCore=%u, extraRows=%u\n"
                "[TilingData] tileCols=%u, xyLocalBytes=%u\n"
                "[TilingData] reduceTmpBytes=%u, tmpLocalBytes=%u\n",
                data.rows, data.cols, data.baseRowsPerCore, data.extraRows, data.tileCols, data.xyLocalBytes,
                data.reduceTmpBytes, data.tmpLocalBytes);
}

// 高级RA把输入视为[R,64]，因此最后一个repeat的无效lane必须补齐。
// Normal逐bit模式：只写尾部，Max填-Inf，Sum填0；有效数据不变。
template <bool isMax>
__aicore__ inline void PadReductionTail(const AscendC::LocalTensor<float>& srcLocal, uint32_t count)
{
    const uint32_t tailElems = count % UB_REPEAT_ELEMS;
    if (tailElems != 0) {
        uint64_t paddingMask[2] = {~((uint64_t{1} << tailElems) - 1), 0};
        AscendC::Duplicate<float, true>(srcLocal[count - tailElems], isMax ? -__builtin_inff() : 0.0f, paddingMask, 1,
                                        1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// FullLoadWithDB和GENERAL共用：RA折半归约到一个repeat，再归约并将结果广播到dstLocal的首32B。
// dstLocal至少256B，首256B都会被改写；仅首32B是有效结果，其余224B不可保留旧值。
// tmpLocal按host侧QueryReduceTmpBytes(count)分配（至少32B），内容会被改写。
// srcLocal容量需补齐到256B，保留count个有效值，只改写尾部padding。
// 三个LocalTensor的起始地址均需32B对齐，所使用的内存区间不能重叠。
template <bool isMax>
__aicore__ inline void ReduceToFirstBlock(const AscendC::LocalTensor<float>& dstLocal,
                                          const AscendC::LocalTensor<float>& srcLocal,
                                          const AscendC::LocalTensor<float>& tmpLocal, uint32_t count)
{
    // 1. srcLocal长度补齐到repeat的整数倍，Max补-Inf，Sum补0。方便reduce完只剩一个repeat。
    PadReductionTail<isMax>(srcLocal, count);
    uint32_t repeatShape[2] = {CeilDiv(count, UB_REPEAT_ELEMS), UB_REPEAT_ELEMS};
    auto tmpBytesLocal = tmpLocal.ReinterpretCast<uint8_t>();

    // 2. 高阶Reduce RA将输入视为[numRepeats,64]，沿repeat轴折半，得到[1,64]的dstLocal。
    // srcInnerPad=true表示内轴已对齐，不负责写padding；isReuseSource=false保留输入有效值。
    if constexpr (isMax) {
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(dstLocal, srcLocal, tmpBytesLocal, repeatShape,
                                                                       true);
    } else {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(dstLocal, srcLocal, tmpBytesLocal, repeatShape,
                                                                       true);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // 3. 对长度为1个repeat的dstLocal做一次wholeReduce，得到一个值放在tmpLocal的第一个位置
    if constexpr (isMax) {
        AscendC::WholeReduceMax<float, true>(
            tmpLocal, dstLocal, UB_REPEAT_ELEMS, 1, 1, 1, 8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE); // ORDER_ONLY_VALUE 表示只需要Max的值不需要Max的位置
    } else {
        AscendC::WholeReduceSum<float, true>(tmpLocal, dstLocal, UB_REPEAT_ELEMS, 1, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    // 4. Brcb将tmpLocal的前8个值分别广播成一个32B block，写满dstLocal的首256B。
    // 只有tmpLocal[0]是归约结果，因此只使用dstLocal的首32B，供后续Vector API按block复用。
    // 全程使用Vector指令，避免GetValue等待V->S同步而打断DB流水。
    AscendC::Brcb(dstLocal, tmpLocal, 1, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
}
