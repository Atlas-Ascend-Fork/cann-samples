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
 * \file matmul_block_mmad_swat.h
 * \brief Block-level  MMAD pipeline for SWAT non-full-load path.
 */

#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../utils/constant.h"
#include "mutex_id.h"

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class TypeA_, class LayoutA_, class TypeB_, class LayoutB_, class TypeC_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, TypeA_, LayoutA_, TypeB_, LayoutB_, TypeC_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithSwat<NO_FULL_LOAD_MODE>, DispatchPolicy_>>> {
public:
    using TypeA = TypeA_;
    using TypeB = TypeB_;
    using TypeC = TypeC_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using L0CType = typename AscendC::Std::conditional<AscendC::IsSameType<TypeA, int8_t>::value, int32_t, float>::type;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA =
        AscendC::IsSameType<LayoutA, asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn>>::value;
    static constexpr bool transB =
        AscendC::IsSameType<LayoutB, asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn>>::value;
    static constexpr int32_t L0C_C0 = 16;
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeA>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeA>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeB>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeB>>>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t kAlign_{1};
    uint64_t kL1Iter_{0};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    constexpr static uint64_t L1_BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT;
    uint64_t bL1Init_{0};
    uint64_t aL1OneBuffer_{1};
    uint64_t bL1OneBuffer_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    bool splitM_{false};
    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    // Custom mmadTrait internal value type
    constexpr static asc::te::mmad_trait MMAD_TRAIT{0, false, false, true, asc::te::mmad_type::normal};
    struct MmadMmTraitConfig { // Custom Trait class
        using TraitType = asc::te::mmad_trait;
        constexpr static const TraitType value = MMAD_TRAIT;
    };

    __aicore__ inline BlockMmad(
        const TupleShape& problemShape, const TupleShape& tileL1Shape, const TupleShape& tileL0Shape, bool l0cDB)
    {
        m_ = asc::te::get<IDX_M_IDX>(problemShape);
        n_ = asc::te::get<IDX_N_IDX>(problemShape);
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        mL1_ = asc::te::get<IDX_M_IDX>(tileL1Shape);
        nL1_ = asc::te::get<IDX_N_IDX>(tileL1Shape);
        kL1_ = asc::te::get<IDX_K_IDX>(tileL1Shape);
        baseM_ = asc::te::get<IDX_M_IDX>(tileL0Shape);
        baseN_ = asc::te::get<IDX_N_IDX>(tileL0Shape);
        baseK_ = asc::te::get<IDX_K_IDX>(tileL0Shape);
        kAlign_ = Align(k_, AscendC::BLOCK_CUBE);
        enableL0cPingPong_ = l0cDB;
        // Non-full load
        aL1OneBuffer_ = mL1_ * kL1_;
        bL1Init_ = aL1OneBuffer_ * L1_BUFFER_NUM;
        bL1OneBuffer_ = nL1_ * kL1_;
        kL1Iter_ = CeilDiv(k_, kL1_); // k_ >= kL1_
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        // ISASI: mutex 初始未锁，首轮 Lock 即过，原预置 SetFlag 已删除。
    }

    __aicore__ inline ~BlockMmad()
    {
    }

    template <typename TensorC, typename TensorA, typename TensorB>
    __aicore__ inline void operator()(TensorC gmC, TensorA gmA, TensorB gmB, const BlockShape& tileShape)
    {
        uint64_t curM = asc::te::get<MNK_M>(tileShape);
        uint64_t curN = asc::te::get<MNK_N>(tileShape);
        uint64_t ml1Align = Align(curM, AscendC::BLOCK_CUBE);
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            // L0C(M<->FIX) 生命周期：M 侧等该 L0C buffer 被 FIX 搬空后占用。
            AscendC::Mutex::Lock<PIPE_M>(L0CMutex(l0cPingPong_ & 0x1));
        }
        kL1_ = Min(k_, kL1_);

        // LoC move out
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType>(l0cOffset), layoutL0C);

        kL1Iter_ = CeilDiv(k_, kL1_); // k_ >= kL1_
        // Loop of k in L1
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;

            uint64_t l1BufId = abL1LoopCnt_ & (L1_BUFFER_NUM - 1);
            uint64_t offsetAl1 = aL1OneBuffer_ * l1BufId * sizeof(TypeA);
            // A/B GM->L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 L1 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId));
            uint64_t offsetBl1 = (bL1Init_ + bL1OneBuffer_ * l1BufId) * sizeof(TypeB);

            // A GM->L1
            auto layoutAL1 = MakeLayoutAL1{}(static_cast<int64_t>(curM), static_cast<int64_t>(curKL1));
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto tensorAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, TypeA>(offsetAl1), layoutAL1);
            // Slice first
            auto gmTileA = gmA.slice(asc::te::make_coord(0, iter0 * kL1_), asc::te::make_shape(curM, curKL1));
            // Copy AL1
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

            // B GM->L1
            auto layoutBL1 = MakeLayoutBL1{}(static_cast<int64_t>(curKL1), static_cast<int64_t>(curN));
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, TypeB>(offsetBl1), layoutBL1);
            // Slice first
            auto gmTileB = gmB.slice(asc::te::make_coord(iter0 * kL1_, 0), asc::te::make_shape(curKL1, curN));
            // Copy BL1
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

            // A/B GM->L1 写完：MTE2 释放，MTE1 占用该 L1（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId));

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            // Loop of k in L0
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 buffer 被 M 用完。
                AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(static_cast<uint16_t>(mte1Flag)));

                // A L1->L0
                auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                auto layoutAL0 =
                    asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeA>>(
                        curM, curK0);
                auto tensorAL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0a, TypeA>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1.slice(asc::te::make_coord(0, iter1 * baseK_), asc::te::make_shape(curM, curK0));
                asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                // B L1->L0
                auto layoutBL0 =
                    asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeB>>(
                        curK0, curN);
                auto tensorBL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, TypeB>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.slice(asc::te::make_coord(iter1 * baseK_, 0), asc::te::make_shape(curK0, curN));
                asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 做 Mmad。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(static_cast<uint16_t>(mte1Flag)));
                AscendC::Mutex::Lock<PIPE_M>(L0Mutex(static_cast<uint16_t>(mte1Flag)));

                // Original mmad parameters
                asc::te::unit_flag_mode unitFlag =
                    enableL0cPingPong_ ?
                        asc::te::unit_flag_mode::disable :
                        ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep);
                bool cmatrixInitVal = (iter0 == 0 && iter1 == 0);
                asc::te::mmad_params mmadParams(curM, curN, curK0, unitFlag, cmatrixInitVal);

                // Pass custom Trait type in mmad
                asc::te::mmad(
                    asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, MmadMmTraitConfig>>{}
                        .with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);

                // Mmad 完成：M 释放该 L0，MTE1 可载下一轮。
                AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(static_cast<uint16_t>(mte1Flag)));
                l0PingPong_++;
            }
            // A/B L1 全部 L1->L0 完成：MTE1 释放该 L1。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(l1BufId));
            abL1LoopCnt_++;
        }
        if (enableL0cPingPong_) {
            // 累加完成：M 释放 L0C，FIX 占用做 L0C->GM 搬运。
            AscendC::Mutex::Unlock<PIPE_M>(L0CMutex(l0cPingPong_ & 0x1));
            AscendC::Mutex::Lock<PIPE_FIX>(L0CMutex(l0cPingPong_ & 0x1));
        }

        // Move data to GM
        asc::te::l0c_to_gm_params fixpParams;
        fixpParams.unit_flag = enableL0cPingPong_ ? asc::te::unit_flag_mode::disable : asc::te::unit_flag_mode::enable_update;
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C);

        if (enableL0cPingPong_) {
            // L0C->GM 完成：FIX 释放该 L0C。
            AscendC::Mutex::Unlock<PIPE_FIX>(L0CMutex(l0cPingPong_ & 0x1));
            l0cPingPong_++;
        }
    }
};
} // namespace Block
