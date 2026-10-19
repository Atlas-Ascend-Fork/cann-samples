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

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "block/block_attn_res_prepare_policy.h"
#include "vf/block_attn_res_prepare_mix_vf.h"
#include "tensor_api/tensor.h"

namespace BlockAttnResPrepareMix {
namespace Epilogue {
namespace Block {

template <class ElementType_ = float, class DispatchPolicy_ = Attention::BlockAttnResPreparePolicy>
class BlockEpilogueBlockAttnResPrepare {
public:
    using ElementType = ElementType_;
    using DispatchPolicy = DispatchPolicy_;

    struct Params {
        GM_ADDR validBlocksGmAddr{nullptr};
        GM_ADDR softmaxMaxGmAddr{nullptr};
        GM_ADDR weightedOutputGmAddr{nullptr};
        GM_ADDR softmaxSumGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t totalD{0U};
        uint32_t baseD{0U};
        uint32_t baseDAlign{0U};
        uint32_t dTileNum{0U};
        uint32_t sAlign{0U};
        uint8_t vUbBufferNum{0U};
        uint64_t eWorkspaceElems{0U};
        uint64_t vUbElems{0U};
        uint64_t dotUbElems{0U};
        uint64_t reduceUbElems{0U};
        uint64_t softmaxUbElems{0U};
        uint64_t workspacePerCoreElems{0U};
        float epsilon{1.0e-6F};
    };

    static_assert(AscendC::Std::is_same_v<ElementType, float>, "BlockAttnResPrepare epilogue only supports FP32.");
    static_assert(
        AscendC::Std::is_same_v<typename DispatchPolicy::ScheduleType, Attention::KernelBlockAttnResPrepareSchedule>,
        "BlockAttnResPrepare epilogue requires its mixed AIC+AIV schedule.");

    __aicore__ inline BlockEpilogueBlockAttnResPrepare()
    {
        if ASCEND_IS_AIV {
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(DEFAULT_EVENT_ID));
        }
    }

    __aicore__ inline ~BlockEpilogueBlockAttnResPrepare()
    {
        if ASCEND_IS_AIV {
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(DEFAULT_EVENT_ID));
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        params_ = &params;
        dotUbOffset_ = static_cast<uint64_t>(params.vUbBufferNum) * params.vUbElems * sizeof(ElementType);
        reduceUbOffset_ = dotUbOffset_ + params.dotUbElems * sizeof(ElementType);
        softmaxUbOffset_ = reduceUbOffset_ + params.reduceUbElems * sizeof(ElementType);
        reciprocalD_ = 1.0F / static_cast<float>(params.totalD);
    }

    template <typename VTensor>
    __aicore__ inline void ReduceV(const VTensor& vTensor)
    {
        ReduceVByDTiles(vTensor);
    }

    template <typename DotTensor, typename EWorkspaceTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline void FinalizeSoftmax(
        const DotTensor& dotTensor, const EWorkspaceTensor& eWorkspaceTensor, const MaxTensor& maxTensor,
        const SumTensor& sumTensor)
    {
        FinalizeSoftmaxRows(dotTensor, eWorkspaceTensor, maxTensor, sumTensor);
    }

    template <typename OutputTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline void ProcessEmptyInput(
        const OutputTensor& outputTensor, const MaxTensor& maxTensor, const SumTensor& sumTensor)
    {
        WriteEmptyOutputs(outputTensor, maxTensor, sumTensor);
    }

private:
    static constexpr uint8_t DEFAULT_EVENT_ID = 0U;
    static constexpr uint32_t SECOND_BUFFER_INDEX = 1U;

    // ------------------------- V reduction stage -------------------------
    template <typename VTensor>
    __aicore__ inline auto MakeVUbTensor(const VTensor& vTensor, uint32_t bufferIndex) const
    {
        const uint64_t byteOffset = static_cast<uint64_t>(bufferIndex) * params_->vUbElems * sizeof(ElementType);
        const int64_t rows = asc::te::get_total_row_shape(vTensor.layout());
        const int64_t columns = asc::te::get_total_column_shape(vTensor.layout());
        return MakeUbTensor<ElementType>(byteOffset, rows, columns, params_->baseDAlign);
    }

    template <typename VTensor>
    __aicore__ inline void LoadVTile(const VTensor& vTensor, uint32_t bufferIndex)
    {
        auto vUbTensor = MakeVUbTensor(vTensor, bufferIndex);
        auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGmToUb, vUbTensor, vTensor);
    }

    template <bool FirstDTile, typename VTensor>
    __aicore__ inline void ReduceVTile(const VTensor& vTensor, uint32_t bufferIndex)
    {
        const int64_t validN = asc::te::get_total_row_shape(vTensor.layout());
        auto vUbTensor = MakeVUbTensor(vTensor, bufferIndex);
        auto sumSquareTensor = MakeUbTensor<ElementType>(reduceUbOffset_, 1, validN, validN);
        BlockAttnResPrepareMix::Epilogue::Tile::ReduceSquare<FirstDTile>::Run(vUbTensor, sumSquareTensor);
    }

    template <typename VTensor>
    __aicore__ inline auto SliceVTile(const VTensor& vTensor, uint32_t dTileIdx) const
    {
        const uint64_t totalD = asc::te::get_total_column_shape(vTensor.layout());
        const uint64_t dOffset = static_cast<uint64_t>(dTileIdx) * params_->baseD;
        const uint64_t remainingD = totalD - dOffset;
        const int64_t validD = static_cast<int64_t>(remainingD < params_->baseD ? remainingD : params_->baseD);
        const int64_t validN = asc::te::get_total_row_shape(vTensor.layout());
        return vTensor.slice(
            asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(dOffset)),
            asc::te::make_shape(validN, validD));
    }

    template <typename VTensor>
    __aicore__ inline void ReduceVByDTiles(const VTensor& vTensor)
    {
        auto firstVTile = SliceVTile(vTensor, 0U);
        LoadVTile(firstVTile, 0U);
        for (uint32_t dTileIdx = 0U; dTileIdx < params_->dTileNum; ++dTileIdx) {
            const uint32_t bufferIndex = dTileIdx % params_->vUbBufferNum;
            WaitMte2ToVector(GetEvent(bufferIndex));
            const uint32_t nextDTileIdx = dTileIdx + 1U;
            if (nextDTileIdx < params_->dTileNum) {
                const uint32_t nextBufferIndex = nextDTileIdx % params_->vUbBufferNum;
                if (nextDTileIdx >= params_->vUbBufferNum) {
                    WaitVectorToMte2(GetEvent(nextBufferIndex));
                }
                auto nextVTile = SliceVTile(vTensor, nextDTileIdx);
                LoadVTile(nextVTile, nextBufferIndex);
            }
            auto currentVTile = SliceVTile(vTensor, dTileIdx);
            if (dTileIdx == 0U) {
                ReduceVTile<true>(currentVTile, bufferIndex);
            } else {
                ReduceVTile<false>(currentVTile, bufferIndex);
            }
        }
        // Complete all outstanding vector reads before the UB ping/pong slots are reused by the next token.
        WaitVectorToMte2(DEFAULT_EVENT_ID);
        if (params_->dTileNum > SECOND_BUFFER_INDEX) {
            WaitVectorToMte2(GetEvent(SECOND_BUFFER_INDEX));
        }
    }

    template <typename Tensor>
    __aicore__ inline static auto SliceRow(const Tensor& tensor, int64_t rowIndex, int64_t columns)
    {
        return tensor.slice(
            asc::te::make_coord(rowIndex, static_cast<int64_t>(0)),
            asc::te::make_shape(static_cast<int64_t>(1), columns));
    }

    // --------------------- RMS and softmax finalization ---------------------
    template <typename DotTensor, typename EWorkspaceTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline void FinalizeSoftmaxRows(
        const DotTensor& dotTensor, const EWorkspaceTensor& eWorkspaceTensor, const MaxTensor& maxTensor,
        const SumTensor& sumTensor)
    {
        const int64_t validSRows = asc::te::get_total_row_shape(dotTensor.layout());
        const int64_t validN = asc::te::get_total_column_shape(dotTensor.layout());
        const int64_t nAlign = asc::te::get_total_column_shape(eWorkspaceTensor.layout());
        if (validSRows == 0) {
            return;
        }
        auto dotUbTensor = MakeUbTensor<ElementType>(dotUbOffset_, validSRows, validN, nAlign);
        auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        // dot/max/sum share UB storage across tokens. Do not overwrite it until the previous UB2GM completes.
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(DEFAULT_EVENT_ID));
        asc::te::copy(copyGmToUb, dotUbTensor, dotTensor);
        WaitMte2ToVector(DEFAULT_EVENT_ID);
        auto sumSquareTensor = MakeUbTensor<ElementType>(reduceUbOffset_, 1, validN, validN);
        auto maxUbTensor = MakeUbTensor<ElementType>(softmaxUbOffset_, validSRows, 1, 1);
        auto sumUbTensor = MakeUbTensor<ElementType>(
            softmaxUbOffset_ + static_cast<uint64_t>(params_->sAlign) * sizeof(ElementType), validSRows, 1, 1);
        for (uint16_t sIndex = 0U; sIndex < static_cast<uint16_t>(validSRows); ++sIndex) {
            auto dotRowTensor = SliceRow(dotUbTensor, static_cast<int64_t>(sIndex), validN);
            auto maxRowTensor = SliceRow(maxUbTensor, static_cast<int64_t>(sIndex), 1);
            auto sumRowTensor = SliceRow(sumUbTensor, static_cast<int64_t>(sIndex), 1);
            BlockAttnResPrepareMix::Epilogue::Tile::RmsSoftmax::Run(
                sumSquareTensor, dotRowTensor, maxRowTensor, sumRowTensor, reciprocalD_, params_->epsilon);
        }
        WaitVectorToMte3(DEFAULT_EVENT_ID);
        auto eUbTensor = MakeUbTensor<ElementType>(dotUbOffset_, validSRows, nAlign, nAlign);
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        asc::te::copy(copyUbToGm, eWorkspaceTensor, eUbTensor);
        asc::te::copy(copyUbToGm, maxTensor, maxUbTensor);
        asc::te::copy(copyUbToGm, sumTensor, sumUbTensor);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(DEFAULT_EVENT_ID));
    }

    // --------------------------- Empty-input stage ---------------------------
    template <typename OutputTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline void WriteEmptyOutputs(
        const OutputTensor& outputTensor, const MaxTensor& maxTensor, const SumTensor& sumTensor)
    {
        constexpr uint64_t OUTPUT_BYTE_OFFSET = 0U;
        const int64_t validS = asc::te::get_total_row_shape(outputTensor.layout());
        const uint64_t totalD = asc::te::get_total_column_shape(outputTensor.layout());
        auto maxUbTensor = MakeUbTensor<ElementType>(softmaxUbOffset_, 1, 1, 1);
        auto sumUbTensor = MakeUbTensor<ElementType>(
            softmaxUbOffset_ + static_cast<uint64_t>(params_->sAlign) * sizeof(ElementType), 1, 1, 1);
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        for (uint16_t sIndex = 0U; sIndex < static_cast<uint16_t>(validS); ++sIndex) {
            BlockAttnResPrepareMix::Epilogue::Tile::InitializeEmptySoftmax::Run(maxUbTensor, sumUbTensor);
            WaitVectorToMte3(DEFAULT_EVENT_ID);
            auto maxRowTensor = SliceRow(maxTensor, static_cast<int64_t>(sIndex), 1);
            auto sumRowTensor = SliceRow(sumTensor, static_cast<int64_t>(sIndex), 1);
            asc::te::copy(copyUbToGm, maxRowTensor, maxUbTensor);
            asc::te::copy(copyUbToGm, sumRowTensor, sumUbTensor);
            WaitMte3ToVector(DEFAULT_EVENT_ID);
            for (uint32_t dTileIdx = 0U; dTileIdx < params_->dTileNum; ++dTileIdx) {
                const uint64_t dOffset = static_cast<uint64_t>(dTileIdx) * params_->baseD;
                const uint64_t remainingD = totalD - dOffset;
                const int64_t validD = static_cast<int64_t>(remainingD < params_->baseD ? remainingD : params_->baseD);
                auto outputUbTensor = MakeUbTensor<ElementType>(OUTPUT_BYTE_OFFSET, 1, validD, params_->baseDAlign);
                Tile::FillUb<ElementType>::FillWithValue(outputUbTensor, 0.0F);
                WaitVectorToMte3(DEFAULT_EVENT_ID);
                auto outputTile = outputTensor.slice(
                    asc::te::make_coord(static_cast<int64_t>(sIndex), static_cast<int64_t>(dOffset)),
                    asc::te::make_shape(static_cast<int64_t>(1), validD));
                asc::te::copy(copyUbToGm, outputTile, outputUbTensor);
                WaitMte3ToVector(DEFAULT_EVENT_ID);
            }
        }
    }

private:
    // ------------------------ Local pipeline events ------------------------
    __aicore__ inline static uint8_t GetEvent(uint32_t bufferIndex)
    {
        return static_cast<uint8_t>(bufferIndex);
    }

    __aicore__ inline static void WaitMte2ToVector(uint8_t eventId)
    {
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(eventId));
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(eventId));
    }

    __aicore__ inline static void WaitVectorToMte2(uint8_t eventId)
    {
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventId));
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventId));
    }

    __aicore__ inline static void WaitVectorToMte3(uint8_t eventId)
    {
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(eventId));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(eventId));
    }

    __aicore__ inline static void WaitMte3ToVector(uint8_t eventId)
    {
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventId));
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventId));
    }

    template <typename T>
    __aicore__ inline static auto MakeUbTensor(uint64_t byteOffset, int64_t rows, int64_t columns, int64_t rowPitch)
    {
        auto shape = asc::te::make_shape(
            asc::te::make_shape(AscendC::Std::Int<1>{}, rows),
            asc::te::make_shape(AscendC::Std::Int<1>{}, columns));
        auto stride = asc::te::make_stride(
            asc::te::make_stride(AscendC::Std::Int<0>{}, rowPitch),
            asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        auto layout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(
            shape, stride);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, T>(byteOffset), layout);
    }

    const Params* __restrict params_{nullptr};
    uint64_t dotUbOffset_{0U};
    uint64_t reduceUbOffset_{0U};
    uint64_t softmaxUbOffset_{0U};
    float reciprocalD_{0.0F};
};

} // namespace Block
} // namespace Epilogue
} // namespace BlockAttnResPrepareMix
