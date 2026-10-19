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
 * \file quant_matmul_mx_block_mmad_k_tail_stepwise_copyout.h
 * \brief MX MMAD pipeline with K-Tail Stepwise Copyout for the SWAT non-full-load path.
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

template <uint64_t FULL_LOAD_MODE_, uint64_t STAGES_>
struct QuantMatmulMxKTailStepwiseCopyout {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
    static constexpr uint64_t stages = STAGES_;
};

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_,
    class LayoutC_>
class BlockMmad<
    DispatchPolicy_, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<DispatchPolicy_, QuantMatmulMxKTailStepwiseCopyout<NO_FULL_LOAD_MODE, 2UL>>>> {
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
    uint64_t l0BufferIndex_{0UL};
    uint64_t l0cBufferIndex_{0UL};
    bool needSplitN_{true};
    bool enableL0cBufferRotation_{false};

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
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params,
        bool enableL0cBufferRotation)
    {
        // Pre-compute all persistent buffer sizes and L1 offsets once per block
        // so the hot path only needs to select pre-computed buffer regions.
        m_ = asc::te::get<IDX_M_IDX>(problemShape);
        n_ = asc::te::get<IDX_N_IDX>(problemShape);
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = asc::te::get<IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
        uint64_t l0cBlockBytes = baseM_ * baseN_ * sizeof(float);
        needSplitN_ = l0cBlockBytes > HALF_L0C_SIZE;
        enableL0cBufferRotation_ = enableL0cBufferRotation && !needSplitN_;
        constexpr uint64_t sizeShift = isDTypeFp4 ? 1UL : 0UL;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        aL1OneBuffer_ = (baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleL1Window_ = scaleKL1_ / kL1_;
        kL1ScaleSize_ = CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleKL1Group_ = CeilDiv(scaleKL1_, MXFP_GROUP_SIZE);
        scaleKL1ScaleSize_ = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|...|A1|B1|AScale1|BScale1|...
        // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|...|A1A3|B1B3|AScale1|BScale1|...
        uint64_t l1HalfSize = AscendC::TOTAL_L1_SIZE >> 1;
#pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            uint64_t l1BufferGroup = bufferId >> 1;
            uint64_t l1HalfOffset = (bufferId & 1UL) * l1HalfSize;
            l1BufferAOffset_[bufferId] = l1HalfOffset + l1BufferGroup * aL1OneBuffer_;
            l1BufferBOffset_[bufferId] =
                l1HalfOffset + L1_BUFFER_GROUP_NUM * aL1OneBuffer_ + l1BufferGroup * bL1OneBuffer_;
        }
#pragma unroll
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * L1_BUFFER_GROUP_NUM;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape)
    {
        // Non-full-load streams both A and B through L1/L0 in chunks. Scale
        // tensors advance in a coarser cadence that matches `scaleKL1_`.
        auto curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        auto curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = enableL0cBufferRotation_ ? (l0cBufferIndex_ & 1) * HALF_L0C_SIZE : 0UL;
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);
        uint64_t scaleWindowIter = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            auto curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            auto curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
            auto curGmAKL1 = curGmBKL1;
            if (scaleWindowIter == 0) {
                // Scale fragments are refreshed only when the current K chunk
                // enters a new scale reuse window.
                // scale-L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 scale L1 buffer 被 MTE1 读完。
                AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));

                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }

                auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
                auto layoutScaleAL1 =
                    asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                        curM, scaleKL1Group_);
                auto tensorScaleAL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto gmBlockScaleA = gmScaleA.slice(
                    asc::te::make_coord(0, kL1Offset / MXFP_GROUP_SIZE),
                    asc::te::make_shape(curM, CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                asc::te::copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

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

            // A/B GM->L1 复用窗口开始：等上一轮该 L1 tile 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId));
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto layoutAL1 = MakeLayoutAL1{}(curM, curPadKL1);
            auto tensorAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto gmBlockA = gmA.slice(asc::te::make_coord(0, kL1Offset), asc::te::make_shape(curM, curGmAKL1));
            ::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = MakeLayoutBL1{}(curPadKL1, curN);
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.slice(asc::te::make_coord(kL1Offset, 0), asc::te::make_shape(curGmBKL1, curN));
            ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);

            // A/B GM->L1 写完：MTE2 释放，MTE1 占用该 L1 tile（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId));

            uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
            // Use the same N partition for first-KL1 initialization and K-tail stepwise copyout.
            bool splitNForCurrentKL1 = needSplitN_ && (iter0 == 0UL || iter0 + 1UL == kL1Iter_);
            uint64_t iterMmadN = splitNForCurrentKL1 ? GetAlignedSplitSize(curN) : curN;
            auto coordScaleKL1 = scaleWindowIter * kL1ScaleSize_;
            auto layoutScaleAL1 =
                asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    curM, scaleKL1ScaleSize_);
            auto tensorScaleAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                layoutScaleAL1);
            auto tensorBlockScaleAL1 = tensorScaleAL1.slice(
                asc::te::make_coord(0, coordScaleKL1), asc::te::make_shape(curM, kL1ScaleSize_));
            auto layoutScaleBL1 =
                asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    scaleKL1ScaleSize_, curN);
            auto tensorScaleBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                layoutScaleBL1);
            for (uint64_t nOffset = 0UL; nOffset < curN; nOffset += iterMmadN) {
                uint64_t curMmadN = (nOffset + iterMmadN > curN) ? (curN - nOffset) : iterMmadN;
                for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    // Finish every K_L0 accumulation for the current N tile
                    // before advancing to the next N tile.
                    auto kL0Offset = iter1 * baseK_;
                    auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
                    bool isFinalAccumulation = (iter0 + 1 == kL1Iter_) && (iter1 + 1 == kL0Iter);
                    uint64_t l0BufId = l0BufferIndex_ & 0x1;
                    uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                    // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 ping-pong 槽被 M 用完。
                    AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(l0BufId));

                    auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                    auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                    auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(
                        curM, curKL0);
                    auto tensorAL0 = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
                    auto tensorBlockAL1 =
                        tensorAL1.slice(asc::te::make_coord(0, kL0Offset), asc::te::make_shape(curM, curKL0));
                    asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

                    auto layoutBL0 = asc::te::make_frame_layout<
                        asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curMmadN);
                    auto tensorBL0 = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
                    auto tensorBlockBL1 = tensorBL1.slice(
                        asc::te::make_coord(kL0Offset, nOffset), asc::te::make_shape(curKL0, curMmadN));
                    asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

                    auto layoutScaleAL0 =
                        asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                            curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                    auto tensorScaleAL0 = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0a, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                    auto CopyL12L0MxScaleA3510 = asc::te::make_copy(::Tile::CopyL12L0MxScaleA3510{});
                    CopyL12L0MxScaleA3510.call(
                        tensorScaleAL0, tensorBlockScaleAL1, asc::te::make_coord(0, kL0Offset));

                    auto layoutScaleBL0 =
                        asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                            CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curMmadN);
                    auto tensorScaleBL0 = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0b, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                    auto tensorBlockScaleBL1 = tensorScaleBL1.slice(
                        asc::te::make_coord(coordScaleKL1, nOffset),
                        asc::te::make_shape(kL1ScaleSize_, curMmadN));
                    auto CopyL12L0MxScaleB3510 = asc::te::make_copy(::Tile::CopyL12L0MxScaleB3510{});
                    CopyL12L0MxScaleB3510.call(
                        tensorScaleBL0, tensorBlockScaleBL1, asc::te::make_coord(kL0Offset, 0));

                    // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 槽做 Mmad。
                    AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(l0BufId));
                    AscendC::Mutex::Lock<PIPE_M>(L0Mutex(l0BufId));
                    asc::te::unit_flag_mode mmadUnitFlag = isFinalAccumulation ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
                    bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                    auto tensorBlockL0C = tensorL0C.slice(
                        asc::te::make_coord(0, nOffset), asc::te::make_shape(curM, curMmadN));
                    asc::te::mmad_params mmadParams;
                    mmadParams.m = static_cast<uint16_t>(curM);
                    mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
                    mmadParams.n = static_cast<uint16_t>(curMmadN);
                    mmadParams.unit_flag = mmadUnitFlag;
                    mmadParams.init_with_zero = mmadCmatrixInitVal;
                    asc::te::mmad(
                        asc::te::mmad_atom<
                            asc::te::mmad_traits<asc::te::mmad_operation, asc::te::MmadTraitMX>>{}
                            .with(mmadParams),
                        tensorBlockL0C, tensorAL0, tensorBL0);
                    if (isFinalAccumulation) {
                        auto gmCBlock =
                            gmC.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(curM, curMmadN));
                        auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
                        asc::te::copy(
                            CopyL0C2GM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), gmCBlock, tensorBlockL0C);
                    }
                    // Mmad 完成：M 释放该 L0 槽，MTE1 可载下一轮。
                    AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(l0BufId));
                    l0BufferIndex_++;
                }
            }

            // Release the current L1 slot only after every L0 slice derived
            // from it has completed its MMAD accumulation.
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

        if (enableL0cBufferRotation_) {
            l0cBufferIndex_++;
        }
    }

private:
    __aicore__ inline uint64_t GetAlignedSplitSize(uint64_t value)
    {
        uint64_t splitSize = CeilAlign(CeilDiv(value, static_cast<uint64_t>(DOUBLE_BUFFER_COUNT)), C0_SIZE);
        return splitSize < value ? splitSize : value;
    }

    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
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
