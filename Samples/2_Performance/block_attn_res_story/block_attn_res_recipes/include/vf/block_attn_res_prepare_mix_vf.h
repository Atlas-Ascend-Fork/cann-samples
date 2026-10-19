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

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

// Prepare-only FP32 epilogue operations for Ascend 950.
// Empty history has numerator=0, logit_max=-FLT_MAX and exp_sum=0.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

namespace BlockAttnResPrepareMix::Epilogue::Tile {

class InitializeEmptySoftmax {
public:
    template <typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(const MaxTensor& maxTensor, const SumTensor& sumTensor)
    {
        using MaxElementType = asc::te::get_attribute_element_type<typename MaxTensor::element_type*>;
        using SumElementType = asc::te::get_attribute_element_type<typename SumTensor::element_type*>;
        using MaxLayoutPattern = asc::te::get_layout_pattern<typename MaxTensor::layout_type>;
        using SumLayoutPattern = asc::te::get_layout_pattern<typename SumTensor::layout_type>;
        static_assert(
            AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
            "InitializeEmptySoftmax only supports FP32 tensors.");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<MaxTensor>, asc::te::location::ub> &&
                AscendC::Std::is_same_v<asc::te::get_mem_location<SumTensor>, asc::te::location::ub>,
            "InitializeEmptySoftmax only supports UB tensors.");
        static_assert(
            AscendC::Std::is_same_v<MaxLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                AscendC::Std::is_same_v<SumLayoutPattern, asc::te::nd_ext_layout_ptn>,
            "InitializeEmptySoftmax requires NDExt tensor layouts.");

        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.data().get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.data().get());
        asc_vf_call<InitializeEmptySoftmaxVf>(maxAddr, sumAddr);
    }

private:
    static constexpr float FP32_LOWEST_FINITE = -__FLT_MAX__;

    static __simd_vf__ inline void InitializeEmptySoftmaxVf(__ubuf__ float* maxAddr, __ubuf__ float* sumAddr)
    {
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
        AscendC::Reg::Duplicate(sumReg, 0.0F, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace BlockAttnResPrepareMix::Epilogue::Tile

namespace BlockAttnResPrepareMix::Epilogue::Tile {

template <bool FirstTile_>
class ReduceSquare {
public:
    template <typename InputTensor, typename SumSquareTensor>
    __aicore__ inline static void Run(const InputTensor& inputTensor, const SumSquareTensor& sumSquareTensor)
    {
        using InputElementType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SumSquareElementType = asc::te::get_attribute_element_type<typename SumSquareTensor::element_type*>;
        using InputLayoutPattern = asc::te::get_layout_pattern<typename InputTensor::layout_type>;
        using SumSquareLayoutPattern = asc::te::get_layout_pattern<typename SumSquareTensor::layout_type>;
        static_assert(
            AscendC::Std::is_same_v<InputElementType, float> && AscendC::Std::is_same_v<SumSquareElementType, float>,
            "ReduceSquare only supports FP32 tensors.");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<InputTensor>, asc::te::location::ub> &&
                AscendC::Std::is_same_v<asc::te::get_mem_location<SumSquareTensor>, asc::te::location::ub>,
            "ReduceSquare only supports UB tensors.");
        static_assert(
            AscendC::Std::is_same_v<InputLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                AscendC::Std::is_same_v<SumSquareLayoutPattern, asc::te::nd_ext_layout_ptn>,
            "ReduceSquare requires NDExt tensor layouts.");

        const uint32_t rowCount = static_cast<uint32_t>(asc::te::get_total_row_shape(inputTensor.layout()));
        const uint32_t validElements = static_cast<uint32_t>(asc::te::get_total_column_shape(inputTensor.layout()));
        const uint32_t rowPitch =
            static_cast<uint32_t>(asc::te::get<1>(asc::te::get<0>(inputTensor.layout().stride())));
        const uint16_t loopCount = static_cast<uint16_t>((validElements + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        auto inputAddr = reinterpret_cast<__ubuf__ float*>(inputTensor.data().get());
        auto sumSquareAddr = reinterpret_cast<__ubuf__ float*>(sumSquareTensor.data().get());
        RunRows(inputAddr, sumSquareAddr, rowCount, validElements, loopCount, rowPitch);
    }

private:
    static constexpr uint32_t FP32_REG_ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    static __simd_vf__ inline void AccumulateSquareVf(
        __ubuf__ float* inputAddr, __ubuf__ float* sumSquareAddr, uint32_t validElements, uint16_t loopCount,
        uint32_t rowIndex)
    {
        AscendC::Reg::RegTensor<float> inputReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> squareAccReg;
        AscendC::Reg::RegTensor<float> squareReduceReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(squareAccReg, 0.0F, allMask);
        uint32_t remaining = validElements;
        for (uint16_t loop = 0U; loop < loopCount; ++loop) {
            AscendC::Reg::MaskReg validMask = AscendC::Reg::UpdateMask<float>(remaining);
            const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(inputReg, inputAddr + offset);
            AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(squareReg, inputReg, inputReg, validMask);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(
                squareAccReg, squareAccReg, squareReg, allMask);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            squareReduceReg, squareAccReg, allMask);
        if constexpr (!FirstTile_) {
            AscendC::Reg::RegTensor<float> previousValueReg;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                previousValueReg, sumSquareAddr + rowIndex);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(
                squareReduceReg, squareReduceReg, previousValueReg, oneMask);
        }
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            sumSquareAddr + rowIndex, squareReduceReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    __aicore__ inline static void RunRows(
        __ubuf__ float* inputAddr, __ubuf__ float* sumSquareAddr, uint32_t rowCount, uint32_t validElements,
        uint16_t loopCount, uint32_t rowPitch)
    {
        for (uint32_t rowIndex = 0U; rowIndex < rowCount; ++rowIndex) {
            asc_vf_call<AccumulateSquareVf>(
                inputAddr + static_cast<uint64_t>(rowIndex) * rowPitch, sumSquareAddr, validElements, loopCount,
                rowIndex);
        }
    }
};

} // namespace BlockAttnResPrepareMix::Epilogue::Tile

namespace BlockAttnResPrepareMix::Epilogue::Tile {

class RmsSoftmax {
public:
    template <typename SumSquareTensor, typename DotTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(
        const SumSquareTensor& sumSquareTensor, const DotTensor& dotTensor, const MaxTensor& maxTensor,
        const SumTensor& sumTensor, float reciprocalD, float epsilon)
    {
        using SumSquareElementType = asc::te::get_attribute_element_type<typename SumSquareTensor::element_type*>;
        using DotElementType = asc::te::get_attribute_element_type<typename DotTensor::element_type*>;
        using MaxElementType = asc::te::get_attribute_element_type<typename MaxTensor::element_type*>;
        using SumElementType = asc::te::get_attribute_element_type<typename SumTensor::element_type*>;
        using SumSquareLayoutPattern = asc::te::get_layout_pattern<typename SumSquareTensor::layout_type>;
        using DotLayoutPattern = asc::te::get_layout_pattern<typename DotTensor::layout_type>;
        using MaxLayoutPattern = asc::te::get_layout_pattern<typename MaxTensor::layout_type>;
        using SumLayoutPattern = asc::te::get_layout_pattern<typename SumTensor::layout_type>;
        static_assert(
            AscendC::Std::is_same_v<SumSquareElementType, float> && AscendC::Std::is_same_v<DotElementType, float> &&
                AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
            "RmsSoftmax only supports FP32 tensors.");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<SumSquareTensor>, asc::te::location::ub> &&
                AscendC::Std::is_same_v<asc::te::get_mem_location<DotTensor>, asc::te::location::ub> &&
                AscendC::Std::is_same_v<asc::te::get_mem_location<MaxTensor>, asc::te::location::ub> &&
                AscendC::Std::is_same_v<asc::te::get_mem_location<SumTensor>, asc::te::location::ub>,
            "RmsSoftmax only supports UB tensors.");
        static_assert(
            AscendC::Std::is_same_v<SumSquareLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                AscendC::Std::is_same_v<DotLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                AscendC::Std::is_same_v<MaxLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                AscendC::Std::is_same_v<SumLayoutPattern, asc::te::nd_ext_layout_ptn>,
            "RmsSoftmax requires NDExt tensor layouts.");

        const uint32_t validN = static_cast<uint32_t>(asc::te::get_total_column_shape(dotTensor.layout()));
        const uint32_t nAlign =
            static_cast<uint32_t>(asc::te::get<1>(asc::te::get<0>(dotTensor.layout().stride())));
        auto sumSquareAddr = reinterpret_cast<__ubuf__ float*>(sumSquareTensor.data().get());
        auto dotAddr = reinterpret_cast<__ubuf__ float*>(dotTensor.data().get());
        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.data().get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.data().get());
        asc_vf_call<RmsSoftmaxVf>(sumSquareAddr, dotAddr, maxAddr, sumAddr, validN, nAlign, reciprocalD, epsilon);
    }

private:
    static __simd_vf__ inline void RmsSoftmaxVf(
        __ubuf__ float* sumSquareAddr, __ubuf__ float* dotAddr, __ubuf__ float* maxAddr, __ubuf__ float* sumAddr,
        uint32_t validN, uint32_t nAlign, float reciprocalD, float epsilon)
    {
        AscendC::Reg::RegTensor<float> sumSquareReg;
        AscendC::Reg::RegTensor<float> dotReg;
        AscendC::Reg::RegTensor<float> rmsReg;
        AscendC::Reg::RegTensor<float> normalizedReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> maxBroadcastReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> expSumReg;
        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t validRemaining = validN;
        uint32_t alignRemaining = nAlign;
        AscendC::Reg::MaskReg validMask = AscendC::Reg::UpdateMask<float>(validRemaining);
        AscendC::Reg::MaskReg alignMask = AscendC::Reg::UpdateMask<float>(alignRemaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumSquareReg, sumSquareAddr);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dotReg, dotAddr);
        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            sumSquareReg, sumSquareReg, reciprocalD, validMask);
        AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            sumSquareReg, sumSquareReg, epsilon, validMask);
        AscendC::Reg::Sqrt<float, AscendC::Reg::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
        AscendC::Reg::Div<float, AscendC::Reg::MaskMergeMode::ZEROING>(normalizedReg, dotReg, rmsReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxReg, normalizedReg, validMask);
        AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxBroadcastReg, maxReg, allMask);
        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(
            expReg, normalizedReg, maxBroadcastReg, validMask);
        AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(expReg, expReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            expSumReg, expReg, validMask);
        AscendC::Reg::Duplicate(zeroReg, 0.0F, allMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, zeroReg, alignMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, expReg, validMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, expSumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace BlockAttnResPrepareMix::Epilogue::Tile

namespace BlockAttnResPrepareMix::Epilogue::Tile {

template <typename T>
class FillUb {
public:
    template <typename Tensor>
    __aicore__ inline static void FillWithValue(const Tensor& dstTensor, T fillValue)
    {
        using TensorElementType = asc::te::get_attribute_element_type<typename Tensor::element_type*>;
        using LayoutPattern = asc::te::get_layout_pattern<typename Tensor::layout_type>;
        static_assert(
            AscendC::Std::is_same_v<TensorElementType, T>,
            "FillUb requires the tensor element type to match the fill value type");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<Tensor>, asc::te::location::ub>,
            "FillUb only supports UB tensors");
        static_assert(
            AscendC::Std::is_same_v<LayoutPattern, asc::te::nd_ext_layout_ptn>,
            "FillUb requires a contiguous NDExt UB tensor");

        auto rowCount = static_cast<uint64_t>(asc::te::get_total_row_shape(dstTensor.layout()));
        auto columnCount = static_cast<uint64_t>(asc::te::get_total_column_shape(dstTensor.layout()));
        auto elementCount = rowCount * columnCount;
        if (elementCount == 0) {
            return;
        }

        constexpr uint32_t elementsPerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        auto repeatTimes =
            (elementCount + static_cast<uint64_t>(elementsPerRepeat) - 1U) / static_cast<uint64_t>(elementsPerRepeat);
        asc_vf_call<FillWithValueVf>(
            (__ubuf__ T*)dstTensor.data().get(), fillValue, static_cast<uint32_t>(elementCount),
            static_cast<uint16_t>(repeatTimes));
    }

private:
    static __simd_vf__ inline void FillWithValueVf(
        __ubuf__ T* dstUbAddr, T fillValue, uint32_t elementCount, uint16_t repeatTimes)
    {
        constexpr uint32_t elementsPerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        AscendC::Reg::RegTensor<T> fillReg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING>(fillReg, fillValue, fullMask);

        uint32_t remainingElements = elementCount;
        for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(remainingElements);
            AscendC::Reg::StoreAlign<T>(
                dstUbAddr + static_cast<uint32_t>(repeatIdx) * elementsPerRepeat, fillReg, mask);
        }
    }
};
} // namespace BlockAttnResPrepareMix::Epilogue::Tile

#endif
