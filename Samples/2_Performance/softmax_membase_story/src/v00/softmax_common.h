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

constexpr int UB_BLOCK_SIZE = 32; // UB 数据块大小，单位为字节。
constexpr int UB_BLOCK_ELEMS = UB_BLOCK_SIZE / static_cast<int>(sizeof(float));
constexpr int UB_REPEAT_SIZE = 256; // 一次 SIMD repeat 处理的字节数。
constexpr int UB_REPEAT_ELEMS = UB_REPEAT_SIZE / static_cast<int>(sizeof(float));

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
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
    GENERAL,
};

struct SoftmaxTilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t tileCols;
    uint32_t xyLocalBytes; // xLocal、yLocal 各自的容量，单位为字节。
    uint32_t tmpLocalBytes;
};

inline void PrintTilingData(const SoftmaxTilingData& data)
{
    std::printf("[TilingData] rows=%u, cols=%u, tileCols=%u\n"
                "[TilingData] xyLocalBytes=%u, tmpLocalBytes=%u\n",
                data.rows, data.cols, data.tileCols, data.xyLocalBytes, data.tmpLocalBytes);
}
