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
 * \file quant_matmul_mx_block_mmad_a_full_load.h
 * \brief Block-level MX MMAD pipeline for the SWAT A-full-load path.
 */

#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../utils/constant.h"
#include "../utils/layout_utils.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/pad_mx_kl1.h"
#include "mutex_id.h"

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_,
    class LayoutC_>
class BlockMmad<
    DispatchPolicy_, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<DispatchPolicy_, QuantMatmulMxMultiBlockWithSwat<A_FULL_LOAD_MODE, 2UL>>>> {
public:
    template <typename T>
    struct TypeUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct TypeUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    template <typename T>
    struct LayoutUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct LayoutUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    using AType = typename TypeUnpack<ATypeTuple_>::Data;
    using ScaleAType = typename TypeUnpack<ATypeTuple_>::Scale;
    using BType = typename TypeUnpack<BTypeTuple_>::Data;
    using ScaleBType = typename TypeUnpack<BTypeTuple_>::Scale;
    using CType = CType_;
    using LayoutA = typename LayoutUnpack<LayoutATuple_>::Data;
    using LayoutScaleA = typename LayoutUnpack<LayoutATuple_>::Scale;
    using LayoutB = typename LayoutUnpack<LayoutBTuple_>::Data;
    using LayoutScaleB = typename LayoutUnpack<LayoutBTuple_>::Scale;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool weightNz = MatmulRecipe::IsWeightNz<LayoutB>::value;
    static constexpr bool transA = MatmulRecipe::IsTrans<LayoutA>::value;
    static constexpr bool transB = MatmulRecipe::IsTrans<LayoutB>::value;
    static constexpr bool isDTypeFp4 =
        AscendC::IsSameType<AType, fp4x2_e1m2_t>::value || AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
    static constexpr uint64_t L1_BUFFER_NUM = DispatchPolicy::stages;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1;
    static constexpr uint64_t L1_BUFFER_GROUP_NUM = L1_BUFFER_NUM >> 1;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr int32_t L0C_C0 = 16;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t MXFP_GROUP_SIZE = 32UL;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t SCALE_BUFFER_NUM = 2;
    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{0UL};
    uint64_t scaleKL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t scaleLoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};

    using MakeLayoutAL1 = asc::te::frame_layout_format<
        AscendC::Std::conditional_t<transA, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL1 = asc::te::frame_layout_format<
        AscendC::Std::conditional_t<transB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        AscendC::Std::Int<C0_SIZE>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
    };

    __aicore__ inline BlockMmad()
    {
        // ISASI: mutex 初始未锁，首轮 Lock 即过，无需预置（原预置 SetFlag 已删除）。
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool enableL0cPingPong)
    {
        // In the A-full-load path, the A tile and its scale stay resident in
        // L1. The offset plan therefore reserves a dedicated resident region
        // and packs the rolling B/scaleB buffers around it.
        m_ = asc::te::get<IDX_M_IDX>(problemShape);
        n_ = asc::te::get<IDX_N_IDX>(problemShape);
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = asc::te::get<IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
        enableL0cPingPong_ = enableL0cPingPong;

        constexpr uint64_t sizeShift = isDTypeFp4 ? 1UL : 0UL;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t mAlign = Align(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
        uint64_t kAlign = Align(k_, MXFP_DIVISOR_SIZE);
        aL1OneBuffer_ = (mAlign * kAlign) >> sizeShift;
        kScaleSize_ = CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleAL1OneBuffer_ = baseM_ * kScaleSize_;
        scaleL1Window_ = scaleKL1_ / kL1_;
        kL1ScaleSize_ = CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleKL1Group_ = CeilDiv(scaleKL1_, MXFP_GROUP_SIZE);
        scaleKL1ScaleSize_ = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // Double-buffered B side: L1 layout is B0|BScale0|A|AScale|...|B1|BScale1|...
        l1BufferAOffset_[0] = bL1OneBuffer_ * L1_BUFFER_GROUP_NUM + scaleBL1OneBuffer_;
        l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
        uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (L1_SIZE >> 1) ?
                                l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ :
                                (L1_SIZE >> 1);
#pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            uint64_t l1BufferGroup = bufferId >> 1;
            uint64_t bHalfOffset = (bufferId & 1UL) == 0 ? 0UL : b1Offset;
            l1BufferBOffset_[bufferId] = bHalfOffset + l1BufferGroup * bL1OneBuffer_;
        }
#pragma unroll
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * L1_BUFFER_GROUP_NUM;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape)
    {
        // This path keeps A-side data resident across the N tiles handled by
        // the same block. The first tile populates the resident A buffers and
        // loads scaleA once; later tiles mainly stream the B side through K.
        auto curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        auto curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);

        auto layoutAL1 = MakeLayoutAL1{}(curM, Align(k_, MXFP_DIVISOR_SIZE));
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[0]), layoutAL1);
        auto layoutScaleAL1 =
            asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curM, kScaleSize_);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[0]), layoutScaleAL1);

        uint64_t scaleWindowIter = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            uint64_t curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
            auto curGmAKL1 = curGmBKL1;

            if (scaleWindowIter == 0) {
                // scaleB follows the reuse window of `scaleKL1_`, so it is
                // refreshed only when the current K chunk enters a new window.
                // scale-L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 scale L1 buffer 被 MTE1 读完。
                AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
                auto layoutScaleBL1 =
                    asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                        scaleKL1Group_, curN);
                auto tensorScaleBL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB.slice(
                    asc::te::make_coord(kL1Offset / MXFP_GROUP_SIZE, 0),
                    asc::te::make_shape(CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                asc::te::copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
                // scale GM->L1 写完：MTE2 释放，MTE1 占用（覆盖整个复用窗口的 scale L1->L0 读）。
                AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
            }

            if (abL1LoopCnt_ == 0) {
                // The resident scaleA tensor is loaded only once for the whole
                // block because the A-full-load path never moves the A tile.
                auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
                auto gmBlockScaleA =
                    gmScaleA.slice(asc::te::make_coord(0, 0), asc::te::make_shape(curM, kScaleSize_));
                asc::te::copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);
            }

            // A/B GM->L1 复用窗口开始：等上一轮该 L1 tile 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId));
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto tensorBlockAL1 =
                tensorAL1.slice(asc::te::make_coord(0, kL1Offset), asc::te::make_shape(curM, curPadKL1));

            if (abL1LoopCnt_ < kL1Iter_) {
                // The resident A buffer covers the full K range; copy the
                // current K slice into its fixed L1 window before B streaming.
                auto gmBlockA =
                    gmA.slice(asc::te::make_coord(0, kL1Offset), asc::te::make_shape(curM, curGmAKL1));
                ::Tile::PadMxKAL1::PadZero(tensorBlockAL1, gmBlockA);
                asc::te::copy(copyGM2L1, tensorBlockAL1, gmBlockA);
            }

            auto layoutBL1 = MakeLayoutBL1{}(curPadKL1, curN);
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.slice(asc::te::make_coord(kL1Offset, 0), asc::te::make_shape(curGmBKL1, curN));
            ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);

            // Both A/B copies for the current L1 slot must complete before the
            // slot is sliced into L0 tiles.
            // A/B GM->L1 写完：MTE2 释放，MTE1 占用该 L1 tile（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId));

            auto tensorBlockScaleAL1 = tensorScaleAL1.slice(
                asc::te::make_coord(0, iter0 * kL1ScaleSize_), asc::te::make_shape(curM, kL1ScaleSize_));
            auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                scaleKL1ScaleSize_, curN);
            auto tensorScaleBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                layoutScaleBL1);
            auto coordScaleKL1 = scaleWindowIter * kL1ScaleSize_;
            auto tensorBlockScaleBL1 = tensorScaleBL1.slice(
                asc::te::make_coord(coordScaleKL1, 0), asc::te::make_shape(kL1ScaleSize_, curN));

            uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                // Split the current B-side L1 chunk into MMAD-sized slices while
                // reusing the resident A-side tensors from the same block.
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
                auto curScaleKL0 = CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                uint64_t l0BufId = l0PingPong_ & 0x1;
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 ping-pong 槽被 M 用完。
                AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(l0BufId));

                auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                auto layoutAL0 =
                    asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0);
                auto tensorAL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1L1 =
                    tensorBlockAL1.slice(asc::te::make_coord(0, kL0Offset), asc::te::make_shape(curM, curKL0));
                asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1L1);

                auto layoutScaleAL0 =
                    asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                        curM, curScaleKL0);
                auto tensorScaleAL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0a, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto CopyL12L0MxScaleA3510 = asc::te::make_copy(::Tile::CopyL12L0MxScaleA3510{});
                CopyL12L0MxScaleA3510.call(tensorScaleAL0, tensorBlockScaleAL1, asc::te::make_coord(0, kL0Offset));

                auto layoutBL0 =
                    asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN);
                auto tensorBL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.slice(asc::te::make_coord(kL0Offset, 0), asc::te::make_shape(curKL0, curN));
                asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

                auto layoutScaleBL0 =
                    asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                        curScaleKL0, curN);
                auto tensorScaleBL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto CopyL12L0MxScaleB3510 = asc::te::make_copy(::Tile::CopyL12L0MxScaleB3510{});
                CopyL12L0MxScaleB3510.call(tensorScaleBL0, tensorBlockScaleBL1, asc::te::make_coord(kL0Offset, 0));

                // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 槽做 Mmad。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(l0BufId));
                AscendC::Mutex::Lock<PIPE_M>(L0Mutex(l0BufId));

                asc::te::unit_flag_mode mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                asc::te::mmad_params mmadParams;
                mmadParams.m = static_cast<uint16_t>(curM);
                mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
                mmadParams.n = static_cast<uint16_t>(curN);
                mmadParams.unit_flag = mmadUnitFlag;
                mmadParams.init_with_zero = mmadCmatrixInitVal;
                asc::te::mmad(
                    asc::te::mmad_atom<
                        asc::te::mmad_traits<asc::te::mmad_operation, asc::te::MmadTraitMX>>{}
                        .with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);
                // Mmad 完成：M 释放该 L0 槽，MTE1 可载下一轮。
                AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(l0BufId));
                l0PingPong_++;
            }

            // A/B L1 tile 全部 L1->L0 完成：MTE1 释放该 L1 tile。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(l1BufId));
            if (scaleWindowIter + 1 == scaleL1Window_ || iter0 == kL1Iter_ - 1) {
                // scale-L1 复用窗口结束：MTE1 释放该 scale L1 buffer。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                scaleLoopCnt_++;
                scaleWindowIter = 0;
            } else {
                ++scaleWindowIter;
            }
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        // The whole block accumulates into one L0C tile, which is flushed once
        // after all K chunks have contributed.
        asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), gmC, tensorL0C);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t kScaleSize_ = 0UL;
    uint64_t scaleL1Window_ = 0UL;
    uint64_t kL1ScaleSize_ = 0UL;
    uint64_t scaleKL1Group_ = 0UL;
    uint64_t scaleKL1ScaleSize_ = 0UL;
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleAOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleBOffset_[SCALE_BUFFER_NUM] = {0UL};
};
} // namespace Block
