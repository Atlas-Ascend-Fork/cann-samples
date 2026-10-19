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
 * \file matmul_block_mmad_streamk.h
 * \brief Block-level  MMAD pipeline for StreamK path.
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
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK, DispatchPolicy_>>> {
public:
    using TypeA = TypeA_;
    using TypeB = TypeB_;
    using TypeC = TypeC_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA =
        AscendC::IsSameType<LayoutA, asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn>>::value;
    static constexpr bool transB =
        AscendC::IsSameType<LayoutB, asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn>>::value;
    static constexpr int32_t L0C_C0 = 16;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 2;
    constexpr static uint64_t L1_BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t bL1Init_{0};
    uint64_t aL1OneBuffer_{1};
    uint64_t bL1OneBuffer_{1};

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeA>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeA>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeB>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeB>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

    // Custom mmadTrait internal value type
    constexpr static asc::te::mmad_trait MMAD_TRAIT{0, false, false, true, asc::te::mmad_type::normal};
    struct MmadMmTraitConfig { // Custom Trait class
        using TraitType = asc::te::mmad_trait;
        constexpr static const TraitType value = MMAD_TRAIT;
    };
    __aicore__ inline BlockMmad(
        const TupleShape& problemShape, const TupleShape& tileL1Shape, const TupleShape& tileL0Shape)
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
        aL1OneBuffer_ = mL1_ * kL1_;
        bL1Init_ = aL1OneBuffer_ * L1_BUFFER_NUM;
        bL1OneBuffer_ = nL1_ * kL1_;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        // ISASI: mutex 初始未锁，首轮 Lock 即过，原预置 SetFlag 已删除。
    }

    __aicore__ inline ~BlockMmad()
    {
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorWorkSpace>
    __aicore__ inline void operator()(
        TensorC gmC, TensorA gmA, TensorB gmB, TensorWorkSpace gmWorkSpace, const BlockShape& tileShape,
        int64_t kCntIndex, bool checkIsSkScene)
    {
        uint64_t curML1 = asc::te::get<MNK_M>(tileShape);
        uint64_t curNL1 = asc::te::get<MNK_N>(tileShape);
        uint64_t curSingleCoreK = asc::te::get<MNK_K>(tileShape);
        uint64_t curKL1Iter = (curSingleCoreK + kL1_ - 1) / kL1_;

        // LoC move out
        auto layoutL0C =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curML1, curNL1);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);
        // Loop of k in L1
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            auto curKL1 = (iter0 + 1 == curKL1Iter) ? (curSingleCoreK - iter0 * kL1_) : kL1_;
            // Switch on pingpong, now only support double buffer in streamk
            uint64_t l1BufId = abL1LoopCnt_ & (L1_BUFFER_NUM - 1);
            uint64_t offsetAL1 = aL1OneBuffer_ * l1BufId * sizeof(TypeA);
            // A GM->L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 A-L1 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId));
            uint64_t offsetBL1 = (bL1Init_ + bL1OneBuffer_ * l1BufId) * sizeof(TypeB);
            // A GM->L1
            auto layoutAL1 = MakeLayoutAL1{}(static_cast<int64_t>(curML1), static_cast<int64_t>(curKL1));
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto tensorAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, TypeA>(offsetAL1), layoutAL1);
            auto gmTileA = gmA.slice(asc::te::make_coord(0, iter0 * kL1_), asc::te::make_shape(curML1, curKL1));
            // Copy AL1
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

            // A GM->L1 写完：MTE2 释放，MTE1 占用该 A-L1（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId));

            // B GM->L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 B-L1 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId + L1_EVENT_ID_OFFSET));
            // B GM->L1
            auto layoutBL1 = MakeLayoutBL1{}(static_cast<int64_t>(curKL1), static_cast<int64_t>(curNL1));
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, TypeB>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.slice(asc::te::make_coord(iter0 * kL1_, 0), asc::te::make_shape(curKL1, curNL1));
            // Copy BL1
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);
            // B GM->L1 写完：MTE2 释放该 B-L1（MTE1 占用延后到首个 L0 切片前）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId + L1_EVENT_ID_OFFSET));

            // Loop of k in L0
            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 buffer 被 M 用完。
                AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(l0PingPong_ & 0x1));
                // A L1->L0
                auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                auto layoutAL0 =
                    asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<TypeA>>(
                        curML1, curK0);
                auto tensorAL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0a, TypeA>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1.slice(asc::te::make_coord(0, iter1 * baseK_), asc::te::make_shape(curML1, curK0));
                asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                if (iter1 == 0) {
                    // B GM->L1 已完成：MTE1 占用该 B-L1（覆盖整段 L1->L0 搬运）。
                    AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId + L1_EVENT_ID_OFFSET));
                }
                // B L1->L0
                auto layoutBL0 =
                    asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<TypeB>>(
                        curK0, curNL1);
                auto tensorBL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, TypeB>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.slice(asc::te::make_coord(iter1 * baseK_, 0), asc::te::make_shape(curK0, curNL1));
                asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 做 Mmad。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(l0PingPong_ & 0x1));
                AscendC::Mutex::Lock<PIPE_M>(L0Mutex(l0PingPong_ & 0x1));

                // Original mmad parameters
                asc::te::unit_flag_mode unitFlag =
                    (iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
                bool cmatrixInitVal = (iter0 == 0 && iter1 == 0);
                asc::te::mmad_params mmadParams(curML1, curNL1, curK0, unitFlag, cmatrixInitVal);
                // Pass custom Trait type in mmad
                asc::te::mmad(
                    asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, MmadMmTraitConfig>>{}
                        .with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);

                // Mmad 完成：M 释放该 L0，MTE1 可载下一轮。
                AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(l0PingPong_ & 0x1));
                l0PingPong_++;
            }
            if (iter0 + 1 == curKL1Iter) {
                auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
                // Depending on checkIsSkScene, decide to move out to GM or WorkSpace
                if (checkIsSkScene) {
                    asc::te::copy(
                        CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmWorkSpace, tensorL0C);
                } else {
                    asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC, tensorL0C);
                }
            }
            // A/B L1 全部 L1->L0 完成：MTE1 释放该 A-L1 / B-L1。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(l1BufId));
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(l1BufId + L1_EVENT_ID_OFFSET));
            abL1LoopCnt_++;
        }
    }
};
} // namespace Block
