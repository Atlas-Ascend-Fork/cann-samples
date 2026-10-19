/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_ub.h
 * \brief Custom copy primitive for packed B4 weight movement from GM to UB.
 */
#pragma once

#include "include/tensor_api/tensor.h"

namespace asc {
namespace te {

// Custom GM-to-UB copy for packed 4-bit weight tensors.
struct CopyGM2UBWeight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();

        uint8_t cacheMode = src.engine().get_cache_mode();

        // get shape and stride
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dstLayout);

        // Extract k1 from shape: ((c0, k1), (n0, n1)).
        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));

        // For packed B4 (2 elements per byte), convert element strides to byte strides via >> 1.
        uint32_t blockLen = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;
        int64_t srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) >> 1;
        int64_t dstStride = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;

        asc_copy_gm2ub_align(
            (__ubuf__ uint8_t*)dst.data().get(), (__gm__ uint8_t*)src.data().get(), blockCount, blockLen, 0, 0, false,
            cacheMode, srcStride, dstStride);
    }
};

} // namespace te
} // namespace asc

// Register copy_traits for the custom GM-to-UB packed-weight copy.
template <>
struct asc::te::copy_traits<asc::te::CopyGM2UBWeight>
    : public asc::te::copy_traits<
          asc::te::CopyGM2UBWeight, asc::te::gm_to_ub_trait_default, asc::te::CopyGM2UBWeight,
          asc::te::gm_to_ub_trait_default> {};
