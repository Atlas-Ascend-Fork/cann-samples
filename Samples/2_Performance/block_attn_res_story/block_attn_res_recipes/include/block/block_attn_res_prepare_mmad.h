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
 * \file block_mmad_matmul_basic.h
 * \brief
 */

#pragma once

#include "tensor_api/tensor.h"

namespace BlockAttnResPrepareMix::Gemm {

constexpr int64_t QUADRUPLE_BUFFER_COUNT = 4LL;
constexpr int32_t MNK_M = 0;
constexpr int32_t MNK_N = 1;
constexpr int32_t MNK_K = 2;
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

// Prepare MM1/MM2 use A/B L1 stages and paired L0 buffers. No bias/BT or quantization scale storage is needed.
template <pipe_t Pipe>
class ScopedSyncLock {
public:
    __aicore__ inline ScopedSyncLock(uint8_t bufferId) : bufferId_(bufferId)
    {
        if ASCEND_IS_AIV {
            if constexpr (Pipe == pipe_t::PIPE_M || Pipe == pipe_t::PIPE_FIX || Pipe == pipe_t::PIPE_MTE1) {
                return;
            }
        }
        asc_lock(Pipe, bufferId_);
    }

    __aicore__ inline ~ScopedSyncLock()
    {
        if ASCEND_IS_AIV {
            if constexpr (Pipe == pipe_t::PIPE_M || Pipe == pipe_t::PIPE_FIX || Pipe == pipe_t::PIPE_MTE1) {
                return;
            }
        }
        asc_unlock(Pipe, bufferId_);
    }

    ScopedSyncLock(const ScopedSyncLock&) = delete;
    ScopedSyncLock& operator=(const ScopedSyncLock&) = delete;

private:
    uint8_t bufferId_;
};

struct BufferSlot {
    uint64_t byteOffset = 0;
    uint8_t bufferId = 0;

    __aicore__ inline uint64_t Addr() const
    {
        return byteOffset;
    }

    template <pipe_t Pipe>
    __aicore__ inline auto Lock() const
    {
        return ScopedSyncLock<Pipe>(bufferId);
    }

    __aicore__ inline auto LockMte2() const
    {
        return Lock<pipe_t::PIPE_MTE2>();
    }
    __aicore__ inline auto LockMte1() const
    {
        return Lock<pipe_t::PIPE_MTE1>();
    }
    __aicore__ inline auto LockM() const
    {
        return Lock<pipe_t::PIPE_M>();
    }
};

class PrepareMmadBuffers {
    static constexpr uint32_t L1_SLOTS = 4;
    static constexpr uint32_t L0_SLOTS = 2;
    // Keep the original L0 lock IDs when removing unused bias/scale slots.
    static constexpr uint8_t L0_EVENT_BASE = 10;
    static constexpr uint8_t L0C_EVENT_BASE = 12;

public:
    PrepareMmadBuffers() = default;

    // L1 初始化（由调用方自行管理 offset 和 bufferId）
    __aicore__ inline void InitAL1(uint32_t idx, uint64_t byteOffset, uint8_t bufferId)
    {
        aL1Slots_[idx] = {byteOffset, bufferId};
    }
    __aicore__ inline void InitBL1(uint32_t idx, uint64_t byteOffset, uint8_t bufferId)
    {
        bL1Slots_[idx] = {byteOffset, bufferId};
    }

    __aicore__ inline void InitL0()
    {
        for (uint32_t i = 0; i < L0_SLOTS; ++i) {
            l0Slots_[i] = {(AscendC::TOTAL_L0A_SIZE / L0_SLOTS) * i, static_cast<uint8_t>(L0_EVENT_BASE + i)};
        }
    }
    __aicore__ inline void InitL0C()
    {
        for (uint32_t i = 0; i < L0_SLOTS; ++i) {
            l0cSlots_[i] = {(AscendC::TOTAL_L0C_SIZE / L0_SLOTS) * i, static_cast<uint8_t>(L0C_EVENT_BASE + i)};
        }
    }

    // Slot 引用访问
    __aicore__ inline const BufferSlot& GetL1ASlot(uint32_t idx) const
    {
        return aL1Slots_[idx];
    }
    __aicore__ inline const BufferSlot& GetL1BSlot(uint32_t idx) const
    {
        return bL1Slots_[idx];
    }
    __aicore__ inline const BufferSlot& GetL0Slot(uint32_t idx) const
    {
        return l0Slots_[idx];
    }
    __aicore__ inline const BufferSlot& GetL0CSlot(uint32_t idx) const
    {
        return l0cSlots_[idx];
    }

private:
    BufferSlot aL1Slots_[L1_SLOTS];
    BufferSlot bL1Slots_[L1_SLOTS];
    BufferSlot l0Slots_[L0_SLOTS];
    BufferSlot l0cSlots_[L0_SLOTS];
};

} // namespace BlockAttnResPrepareMix::Gemm

namespace BlockAttnResPrepareMix {
namespace Gemm {
namespace Block {

// The sample uses FP32 ND x batched-DN (MM1) and FP32 ND x ND (MM2), without bias or quantization.
template <bool BATCHED_B>
class BlockAttnResPrepareMmad {
public:
    using AType = float;
    using BType = float;
    using CType = float;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<BATCHED_B, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TripleShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = false;
    static constexpr bool TRANS_B = BATCHED_B;
    // AL1 Layout
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;
    // BL1 Layout
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    // kernel params
    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
    };

public:
    __aicore__ inline BlockAttnResPrepareMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockAttnResPrepareMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        l1Stages_ = params.l1Stages;
        enableL0cPingPong_ = params.l0cStages > 1;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        uint64_t aL1OneSize = mL1_ * kL1_ * sizeof(AType);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        uint64_t stride = QUADRUPLE_BUFFER_COUNT / l1Stages_;
        // Each active L1 stage stores A followed by B; stages retain the original slot spacing.
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            uint64_t base = slotSize * stride * i;
            bufMgr_.InitAL1(i, base, i);
            bufMgr_.InitBL1(i, base + aL1OneSize, i);
        }
        bufMgr_.InitL0();
        bufMgr_.InitL0C();
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorC& gmC, TupleShape& blockShape)
    {
        // m0 == m1 && n0 == n1
        int64_t curM =
            BlockAttnResPrepareMix::Gemm::Min(asc::te::get<MNK_M>(blockShape), static_cast<int64_t>(baseM_));
        int64_t curN =
            BlockAttnResPrepareMix::Gemm::Min(asc::te::get<MNK_N>(blockShape), static_cast<int64_t>(baseN_));
        uint64_t oriK = asc::te::get<MNK_K>(blockShape); // 非全载blockShape的K维度固定返回原始K
        const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cPingPong_ & 0x1);
        // LoC搬出
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(curM, curN);
        auto tensorL0C = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot.Addr()), layoutL0C);

        kL1_ = BlockAttnResPrepareMix::Gemm::Min(oriK, kL1_);
        kL1Iter_ = BlockAttnResPrepareMix::Gemm::CeilDiv(oriK, kL1_);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (oriK - kL1_ * iter0) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);

            const auto& aL1Slot = bufMgr_.GetL1ASlot(l1BufId);
            const auto& bL1Slot = bufMgr_.GetL1BSlot(l1BufId);

            // GM->L1
            TripleShape l1Shape{curM, curN, static_cast<int64_t>(curKL1)};
            auto l1Slots = AscendC::Std::make_tuple(aL1Slot, bL1Slot);
            auto l1TensorTuple = CopyL1FromGM(gmA, gmB, l1Shape, l1Slots, iter0);
            auto tensorAL1 = asc::te::get<0>(l1TensorTuple);
            auto tensorBL1 = asc::te::get<1>(l1TensorTuple);

            ComputeL0Tiles(tensorAL1, tensorBL1, tensorL0C, l1Slots, curM, curN, curKL1, iter0);
            abL1LoopCnt_++;
        }

        // 数据搬出到GM
        CopyL0CToGM(gmC, tensorL0C);

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    template <typename TensorA, typename TensorB, typename TensorC, typename SlotsTuple>
    __aicore__ inline void ComputeL0Tiles(
        TensorA& tensorAL1, TensorB& tensorBL1, TensorC& tensorL0C, const SlotsTuple& l1Slots, int64_t curM,
        int64_t curN, uint64_t curKL1, uint64_t iter0)
    {
        uint64_t kL0Iter = BlockAttnResPrepareMix::Gemm::CeilDiv(curKL1, baseK_);
        for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
            const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

            // A L1->L0
            TripleShape l0Shape{curM, curN, static_cast<int64_t>(curK0)};
            auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, l0Shape, l0Slot, baseK_ * iter1, l1Slots);
            auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
            auto tensorBL0 = asc::te::get<1>(l0TensorTuple);

            bool initCmatrix = iter0 == 0 && iter1 == 0;
            asc::te::unit_flag_mode unitFlag =
                ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep);

            {
                auto l0Lock = l0Slot.LockM();
                Compute(tensorAL0, tensorBL0, tensorL0C, l0Shape, unitFlag, initCmatrix);
            }
            l0PingPong_++;
        }
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAToL1(
        const TensorA& tensorA, const BufferSlot& aL1Slot, uint64_t curML1, uint64_t curKL1, uint64_t kIdx)
    {
        // A GM->L1
        auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Slot.Addr()), layoutAL1);
        {
            auto lock = aL1Slot.LockMte2();
            auto gmTileA =
                tensorA.slice(asc::te::make_coord(0, kIdx * kL1_), asc::te::make_shape(curML1, curKL1));
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);
        }

        return tensorAL1;
    }

    template <typename TensorA, typename TensorB, typename SlotsTuple>
    __aicore__ inline auto CopyL1FromGM(
        const TensorA& tensorA, const TensorB& tensorB, const TripleShape& l1Shape, const SlotsTuple& slotsTuple,
        uint64_t kIdx)
    {
        uint64_t curML1 = asc::te::get<MNK_M>(l1Shape);
        uint64_t curNL1 = asc::te::get<MNK_N>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);
        const auto& aL1Slot = asc::te::get<0>(slotsTuple);
        const auto& bL1Slot = asc::te::get<1>(slotsTuple);

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto tensorAL1 = CopyAToL1(tensorA, aL1Slot, curML1, curKL1, kIdx);

        // B GM->L1
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Slot.Addr()), layoutBL1);
        {
            auto lock = bL1Slot.LockMte2();
            if constexpr (BATCHED_B) {
                CopyBatchedBToConcatL1(copyGM2L1, tensorBL1, tensorB, bL1Slot.Addr(), curKL1, kIdx);
            } else {
                auto gmTileB =
                    tensorB.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curNL1));
                asc::te::copy(copyGM2L1, tensorBL1, gmTileB);
            }
        }
        return AscendC::Std::make_tuple(tensorAL1, tensorBL1);
    }

    template <typename TensorBL1, typename TensorB>
    __aicore__ inline void ValidateBatchedBTypes()
    {
        using TensorBElementType = asc::te::get_attribute_element_type<typename TensorB::element_type*>;
        using TensorBLayoutPattern = asc::te::get_layout_pattern<typename TensorB::layout_type>;
        using TensorBL1ElementType = asc::te::get_attribute_element_type<typename TensorBL1::element_type*>;
        using TensorBL1LayoutPattern = asc::te::get_layout_pattern<typename TensorBL1::layout_type>;
        using ExpectedBL1LayoutPattern =
            AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        static_assert(
            AscendC::Std::is_same_v<TensorBElementType, BType>,
            "Batched-B GM tensor element type must match BlockMmad BType.");
        static_assert(
            AscendC::Std::is_same_v<TensorBL1ElementType, BType>,
            "Batched-B L1 tensor element type must match BlockMmad BType.");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<TensorB>, asc::te::location::gm>,
            "Batched-B source tensor must reside in GM.");
        static_assert(
            AscendC::Std::is_same_v<asc::te::get_mem_location<TensorBL1>, asc::te::location::l1>,
            "Batched-B destination tensor must reside in L1.");
        static_assert(
            AscendC::Std::is_same_v<TensorBLayoutPattern, LayoutB>,
            "Batched-B GM tensor layout pattern must match BlockMmad LayoutB.");
        static_assert(
            AscendC::Std::is_same_v<TensorBL1LayoutPattern, ExpectedBL1LayoutPattern>,
            "Batched-B L1 tensor layout must match the BlockMmad transpose mode.");
        static_assert(
            TensorB::layout_type::depth == asc::te::five_dim_data,
            "Batched-B source tensor must contain an outer batch and a two-dimensional matrix layout.");
    }

    template <typename CopyGM2L1, typename TensorBL1, typename TensorB>
    __aicore__ inline void CopyBatchedBToConcatL1(
        const CopyGM2L1& copyGM2L1, const TensorBL1& tensorBL1, const TensorB& tensorB, uint64_t bL1Address,
        uint64_t curKL1, uint64_t kIdx)
    {
        ValidateBatchedBTypes<TensorBL1, TensorB>();

        auto gmBLayout = tensorB.layout();
        const uint64_t batchCount = static_cast<uint64_t>(asc::te::get<0>(gmBLayout.shape()));
        using GmBLayoutPattern = asc::te::get_layout_pattern<typename TensorB::layout_type>;
        using BLayoutTrait = asc::te::layout_trait_default<BType>;
        auto singleGmBLayout = asc::te::make_pattern_layout<GmBLayoutPattern, BLayoutTrait>(
            asc::te::get<1>(gmBLayout.shape()), asc::te::get<1>(gmBLayout.stride()));
        const uint64_t singleN = asc::te::get_total_column_shape(singleGmBLayout);
        auto gmTileB = tensorB.slice(
            asc::te::make_coord(0UL, asc::te::make_coord(kIdx * kL1_, 0UL)),
            asc::te::make_shape(batchCount, asc::te::make_shape(curKL1, singleN)));

        // Re-view the same BL1 storage as a batch so one GM2L1 copy packs the matrices into adjacent N ranges.
        // The original two-dimensional tensorBL1 remains the compute view used by L1-to-L0 and MMAD.
        auto concatBL1Layout = tensorBL1.layout();
        auto singleBL1 = tensorBL1.slice(asc::te::make_coord(0UL, 0UL), asc::te::make_shape(curKL1, singleN));
        auto singleBL1Layout = singleBL1.layout();
        const uint64_t batchStride = concatBL1Layout(asc::te::make_coord(0UL, singleN));
        using BL1Layout = typename TensorBL1::layout_type;
        using BL1LayoutPattern = asc::te::get_layout_pattern<BL1Layout>;
        auto batchedBL1Layout = asc::te::make_pattern_layout<BL1LayoutPattern, BLayoutTrait>(
            asc::te::make_shape(batchCount, singleBL1Layout.shape()),
            asc::te::make_stride(batchStride, singleBL1Layout.stride()));
        auto batchedBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Address), batchedBL1Layout);
        asc::te::copy(copyGM2L1, batchedBL1, gmTileB);
    }

    template <typename TensorA, typename TensorB, typename SlotsTuple>
    __aicore__ inline auto CopyL0FromL1(
        const TensorA& tensorAL1, const TensorB& tensorBL1, const TripleShape& l0Shape, const BufferSlot& l0Slot,
        uint64_t kIdx, const SlotsTuple& slotsTuple)
    {
        auto curM0 = asc::te::get<MNK_M>(l0Shape);
        auto curN0 = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        const auto& aL1Slot = asc::te::get<0>(slotsTuple);
        const auto& bL1Slot = asc::te::get<1>(slotsTuple);
        // A L1->L0A
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM0, curK0);
        auto tensorAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Slot.Addr()), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kIdx), asc::te::make_shape(curM0, curK0));
        {
            auto l1LockA = aL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        }

        // B L1->L0B
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            curK0, curN0);
        auto tensorBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Slot.Addr()), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kIdx, 0), asc::te::make_shape(curK0, curN0));
        {
            auto l1LockB = bL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        }

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyL0CToGM(TensorC& gmC, TensorL0C& tensorL0C)
    {
        asc::te::l0c_to_gm_params fixpParams{asc::te::unit_flag_mode::enable_update};
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C);
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void Compute(
        const TensorA& tensorAL0, const TensorB& tensorBL0, TensorC& tensorL0C, const TripleShape& l0Shape,
        asc::te::unit_flag_mode unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM0 = asc::te::get<MNK_M>(l0Shape);
        auto curN0 = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        // Mmad参数
        asc::te::mmad_params mmadParams{
            static_cast<uint16_t>(curM0), static_cast<uint16_t>(curN0), static_cast<uint16_t>(curK0), unitFlag,
            initCmatrix};
        asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
    }

private:
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    uint64_t kL1Iter_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    // 全流水线Buffer管理器, <MaxL1ASlots = 4, MaxL1BSlots = 4, MaxL0Slots = 2>
    PrepareMmadBuffers bufMgr_;
};
} // namespace Block
} // namespace Gemm
} // namespace BlockAttnResPrepareMix
