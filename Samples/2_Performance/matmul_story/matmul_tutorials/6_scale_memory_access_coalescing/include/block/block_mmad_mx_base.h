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

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../utils/quant_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"

namespace Block {
using namespace AscendC;

// ISASI Mutex ID 分段：L1(MTE2<->MTE1): [0,8) | L0(MTE1<->M): [8,16)。
// 本文件 L0C 由 mmadParams.unit_flag 硬件同步，无 M<->FIX 软件边，故不需 L0CMutex。
__aicore__ inline uint8_t L1Mutex(uint16_t id) { return static_cast<uint8_t>(id); }
__aicore__ inline uint8_t L0Mutex(uint16_t id) { return static_cast<uint8_t>(id + 8); }

template <class AType_, class BType_, class CType_>
class BlockMmadMx {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = false;
    static constexpr bool transB = true;
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr int32_t L0C_C0 = 16;

    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t MXFP_GROUP_SIZE_LOCAL = 32UL;
    static constexpr uint64_t MXFP_DIVISOR_SIZE_LOCAL = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE_LOCAL = 2UL;
    static constexpr uint64_t SCALE_BUFFER_NUM = 2;

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL1 = asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};
        uint64_t scaleKL1{0};
        uint64_t l1BufNum{2};
    };

    __aicore__ inline BlockMmadMx()
    {
        // ISASI: mutex 初始未锁，首轮 Lock 即过，无需预置（原预置 SetFlag 已删除）。
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params)
    {
        m_ = asc::te::get<IDX_M_IDX>(problemShape);
        n_ = asc::te::get<IDX_N_IDX>(problemShape);
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = asc::te::get<IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
        l1BufNum_ = l1Params.l1BufNum;
        scaleKL1Ratio_ = scaleKL1_ / kL1_;

        bL1OneBuffer_ = (baseN_ * kL1_) >> 1;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;

        aL1OneBuffer_ = (baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE_LOCAL)) >> 1;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
        constexpr uint64_t L1_HALF_BYTES = L1_SIZE / 2;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            uint64_t halfOffset = L1_HALF_BYTES * (bufferId & 1);
            l1BufferAOffset_[bufferId] = halfOffset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] =
                halfOffset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] =
                l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
            l1BufferScaleBOffset_[bufferId] =
                l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }

        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorAL1, typename TensorScaleAL1, typename TensorBL1, typename TensorScaleBL1, typename TensorL0C>
    __aicore__ inline void IterateL0(
        TensorAL1 tensorBlockAL1, TensorScaleAL1 tensorBlockScaleAL1,
        TensorBL1 tensorBL1, TensorScaleBL1 tensorBlockScaleBL1,
        uint64_t curM, uint64_t curN,
        uint64_t curGmBKL1, uint64_t curPadKL1, uint64_t iter0,
        uint64_t scaleKL1IterOffset,
        TensorL0C& tensorL0C)
    {
        uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            auto kL0Offset = iter1 * baseK_;
            auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
            auto curScaleKL0 = CeilDiv(curKL0, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 ping-pong 槽被 M 用完。
            AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(l0PingPong_ & 0x1));

            auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
            auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0);
            auto tensorAL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1Slice = tensorBlockAL1.slice(asc::te::make_coord(0, kL0Offset), asc::te::make_shape(curM, curKL0));
            asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1Slice);

            auto layoutScaleAL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curM, curScaleKL0);
            auto tensorScaleAL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0a, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
            auto CopyL12L0MxScaleA = asc::te::make_copy(::Tile::CopyL12L0MxScaleA3510{});
            CopyL12L0MxScaleA.call(tensorScaleAL0, tensorBlockScaleAL1, asc::te::make_coord(0, scaleKL1IterOffset + kL0Offset));

            auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN);
            auto tensorBL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1Slice = tensorBL1.slice(asc::te::make_coord(kL0Offset, 0), asc::te::make_shape(curKL0, curN));
            asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1Slice);

            auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curScaleKL0, curN);
            auto tensorScaleBL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0b, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
            auto CopyL12L0MxScaleB = asc::te::make_copy(::Tile::CopyL12L0MxScaleB3510{});
            CopyL12L0MxScaleB.call(tensorScaleBL0, tensorBlockScaleBL1, asc::te::make_coord(scaleKL1IterOffset + kL0Offset, 0));

            // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 槽做 Mmad。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(l0PingPong_ & 0x1));
            AscendC::Mutex::Lock<PIPE_M>(L0Mutex(l0PingPong_ & 0x1));

            asc::te::unit_flag_mode mmadUnitFlag =
                (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter)
                    ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
            bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
asc::te::mmad_params mmadParams;
mmadParams.m = static_cast<uint16_t>(curM);
mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE_LOCAL));
mmadParams.n = static_cast<uint16_t>(curN);
mmadParams.unit_flag = mmadUnitFlag;
mmadParams.init_with_zero = mmadCmatrixInitVal;
asc::te::mmad(
    asc::te::mmad_atom<
        asc::te::mmad_traits<asc::te::mmad_operation, asc::te::MmadTraitMX>>{}
        .with(mmadParams),
                tensorL0C, tensorAL0, tensorBL0);

            // Mmad 完成：M 释放该 L0 槽，MTE1 可载下一轮。
            AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(l0PingPong_ & 0x1));
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC,
        BlockShape singleShape)
    {
        auto curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        auto curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            uint64_t curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE_LOCAL);
            auto curGmAKL1 = curGmBKL1;

            if (iter0 % scaleKL1Ratio_ == 0) {
                uint64_t scaleKL1Offset = iter0 * kL1_;
                // scale-L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 scale L1 buffer 被 MTE1 读完。
                AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
                uint64_t curScaleKL1 = scaleKL1_;
                if (scaleKL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - scaleKL1Offset;
                }
                auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    curM, CeilDiv(scaleKL1_, MXFP_GROUP_SIZE_LOCAL));
                auto tensorScaleAL1Buf = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(0, scaleKL1Offset / MXFP_GROUP_SIZE_LOCAL),
                    asc::te::make_shape(
                        curM, CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL));
                asc::te::copy(CopyScaleGM2L1, tensorScaleAL1Buf, gmBlockScaleA);

                auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    CeilDiv(scaleKL1_, MXFP_GROUP_SIZE_LOCAL), curN);
                auto tensorScaleBL1Buf = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(scaleKL1Offset / MXFP_GROUP_SIZE_LOCAL, 0),
                    asc::te::make_shape(
                        CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL, curN));
                asc::te::copy(CopyScaleGM2L1, tensorScaleBL1Buf, gmBlockScaleB);
                // scale GM->L1 写完：MTE2 释放，MTE1 占用（覆盖整个复用窗口的 scale L1->L0 读）。
                AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
            }

            // A/B GM -> L1
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(l1BufId));
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

            auto layoutAL1Iter = MakeLayoutAL1{}(curM, curGmAKL1);
            auto tensorAL1Iter = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1Iter);
            auto gmBlockA = gmA.slice(asc::te::make_coord(0, kL1Offset), asc::te::make_shape(curM, curGmAKL1));
            asc::te::copy(copyGM2L1, tensorAL1Iter, gmBlockA);

            auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
            auto tensorBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.slice(asc::te::make_coord(kL1Offset, 0), asc::te::make_shape(curGmBKL1, curN));
            asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);

            // A/B GM->L1 写完：MTE2 释放，MTE1 占用该 L1 tile（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(l1BufId));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(l1BufId));

            uint64_t scaleKL1IterOffset = (iter0 % scaleKL1Ratio_) * kL1_;

            auto layoutScaleAL1ForL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                curM, CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL);
            auto tensorBlockScaleAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                layoutScaleAL1ForL0);

            auto layoutScaleBL1ForL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL, curN);
            auto tensorBlockScaleBL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                layoutScaleBL1ForL0);

            IterateL0(tensorAL1Iter, tensorBlockScaleAL1,
                      tensorBL1, tensorBlockScaleBL1,
                      curM, curN, curGmBKL1, curPadKL1, iter0,
                      scaleKL1IterOffset, tensorL0C);

            // A/B L1 tile 全部 L1->L0 完成：MTE1 释放该 L1 tile。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(l1BufId));
            if ((iter0 + 1) % scaleKL1Ratio_ == 0 || iter0 == kL1Iter_ - 1) {
                // scale-L1 复用窗口结束：MTE1 释放该 scale L1 buffer。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(SCALE_BUFFER_FLAG_0 + scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), gmC, tensorL0C);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_;
    uint64_t baseN_;
    uint64_t baseK_;
    uint64_t kL1_;
    uint64_t scaleKL1_;
    uint64_t l1BufNum_;
    uint64_t kL1Iter_;
    uint64_t scaleKL1Ratio_;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t l1BufferScaleAOffset_[2] = {0UL};
    uint64_t l1BufferScaleBOffset_[2] = {0UL};
};
} // namespace Block
