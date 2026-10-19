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

#include "include/tensor_api/tensor.h"

namespace asc {
namespace te {

// Custom UB to L1 copy class for 8-bit weight
// Both UB and L1 use NZ (fractal) format
struct CopyUB2L1Custom {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using type = typename U::element_type;
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();

        // get shape and stride tuples
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);

        // Extract dimensions from srcShape = ((c0, k1), (n0, n1))
        // Std::get<0>(srcShape) = (c0, k1), Std::get<1>(srcShape) = (n0, n1)
        uint16_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        uint16_t k1 = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint16_t n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        uint16_t n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));

        // Extract innerStride from srcStride = ((1, n1*InnerStride), (c0, InnerStride))
        // For UB2L1, use column stride (InnerStride) not row stride (n1*InnerStride)
        int64_t innerStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));

        // Total number of fractal blocks to copy
        uint16_t blockCount = k1 * n1;

        // Block length in 32B units
        uint32_t blockLen = (n0 * c0 * sizeof(type)) / 32;

        // Source stride in 32B units
        int64_t srcStride = (innerStride * sizeof(type)) / 32 - blockLen;

        // Destination stride in 32B units (contiguous in L1)
        int64_t dstStride = 0;

        asc_copy_ub2l1(
            (__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace te
} // namespace asc

// Register copy_traits with standard trait
template <>
struct asc::te::copy_traits<asc::te::CopyUB2L1Custom>
    : public asc::te::copy_traits<
          asc::te::CopyUB2L1Custom, asc::te::ub_to_l1_trait_default,
          asc::te::CopyUB2L1Custom, asc::te::ub_to_l1_trait_default> {};

