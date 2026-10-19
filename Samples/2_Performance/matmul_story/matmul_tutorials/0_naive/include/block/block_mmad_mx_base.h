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
#include "../../../common/kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../utils/quant_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"

namespace Block {

// ISASI Mutex ID 分段：L1(MTE2<->MTE1): [0,8) | L0(MTE1<->M): [8,16) | L0C(M<->FIX): [16,24)。
// 本文件为甲分离式，L0C 有显式 M<->FIX 软件边，故需 L0CMutex。
__aicore__ inline uint8_t L1Mutex(uint16_t id)  { return static_cast<uint8_t>(id); }
__aicore__ inline uint8_t L0Mutex(uint16_t id)  { return static_cast<uint8_t>(id + 8); }
__aicore__ inline uint8_t L0CMutex(uint16_t id) { return static_cast<uint8_t>(id + 16); }

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

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};
    };

    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{128};
    uint64_t kL1_{1024};
    uint64_t kL1TileNum;
    uint64_t tailKL1;

    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<IDX_M_IDX>(problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<IDX_N_IDX>(problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<IDX_K_IDX>(problemShape));
        baseM_ = static_cast<uint64_t>(asc::te::get<IDX_M_IDX>(l0TileShape));
        baseN_ = static_cast<uint64_t>(asc::te::get<IDX_N_IDX>(l0TileShape));
        baseK_ = static_cast<uint64_t>(asc::te::get<IDX_K_IDX>(l0TileShape));
        if (baseK_ == 0) {
            baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        }
        kL1_ = (l1Params.kL1 > 0) ? l1Params.kL1 : (baseK_ * 4);
        kL1TileNum = CeilDiv(k_, kL1_);
        tailKL1 = k_ - ((kL1TileNum - 1) * kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA aGlobal,
        TensorB bGlobal,
        TensorScaleA scaleAGlobal,
        TensorScaleB scaleBGlobal,
        TensorC cGlobal,
        const BlockShape& singleShape)
    {
        uint64_t curM = static_cast<uint64_t>(asc::te::get<IDX_M_TILEIDX>(singleShape));
        uint64_t curN = static_cast<uint64_t>(asc::te::get<IDX_N_TILEIDX>(singleShape));

        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(curM, curN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);

        // ISASI: mutex 初始未锁，首轮 Lock 即过，原预置 SetFlag 已删除。

        uint64_t offsetA = 0;
        uint64_t offsetB = 0;
        uint64_t offsetScaleA = 0;
        uint64_t offsetScaleB = 0;

        // L0C(M<->FIX) 生命周期：M 侧等该 L0C buffer 被 FIX 搬空后占用。
        AscendC::Mutex::Lock<PIPE_M>(L0CMutex(0));
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t curKL1 = (iter0 == (kL1TileNum - 1)) ? tailKL1 : kL1_;

            // A/B/scale GM->L1(MTE2<->MTE1) 复用窗口开始：等上一轮该 L1 被 MTE1 读完。
            AscendC::Mutex::Lock<PIPE_MTE2>(L1Mutex(0));
            uint64_t offsetAL1 = 0;
            uint64_t offsetBL1 = baseM_ * kL1_ / OFFSET_4_8;
            uint64_t offsetScaleAL1 = (baseM_ + baseN_) * kL1_ / OFFSET_4_8;
            uint64_t offsetScaleBL1 = (baseM_ + baseN_) * kL1_ / OFFSET_4_8 + baseM_ * CeilDiv(kL1_, MXFP_GROUP_SIZE);
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto layoutAL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL1);
            auto tensorAL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAL1), layoutAL1);
            auto gmBlockA = aGlobal.slice(asc::te::make_coord(0, offsetA),
                asc::te::make_shape(curM, curKL1));
            asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL1, curN);
            auto tensorBL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBL1), layoutBL1);
            auto gmBlockB = bGlobal.slice(asc::te::make_coord(offsetB, 0),
                asc::te::make_shape(curKL1, curN));
            asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);
            auto gmBlockScaleA = scaleAGlobal.slice(asc::te::make_coord(0, offsetScaleA),
                asc::te::make_shape(
                    curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
            auto gmBlockScaleB = scaleBGlobal.slice(asc::te::make_coord(offsetScaleB, 0),
                asc::te::make_shape(
                    CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
            auto copyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(offsetScaleAL1), layoutScaleAL1);
            asc::te::copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
            auto tensorScaleBL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(offsetScaleBL1), layoutScaleBL1);
            asc::te::copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            offsetA += curKL1;
            offsetScaleA += CeilDiv(curKL1, MXFP_GROUP_SIZE);
            offsetB += curKL1;
            offsetScaleB += CeilDiv(curKL1, MXFP_GROUP_SIZE);
            // GM->L1 写完：MTE2 释放，MTE1 占用该 L1（覆盖整段 L1->L0 搬运）。
            AscendC::Mutex::Unlock<PIPE_MTE2>(L1Mutex(0));
            AscendC::Mutex::Lock<PIPE_MTE1>(L1Mutex(0));

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK_);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = (iter1 == (kL0TileNum - 1)) ? tailKL0 : baseK_;

                // L0(MTE1<->M) 生命周期：MTE1 侧等该 L0 buffer 被 M 用完。
                AscendC::Mutex::Lock<PIPE_MTE1>(L0Mutex(0));
                uint64_t l0Offset = 0;
                uint64_t kL0Offset = iter1 * baseK_;
                auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0);
                auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kL0Offset), asc::te::make_shape(curM, curKL0));
                asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN);
                auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kL0Offset, 0), asc::te::make_shape(curKL0, curN));
                asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                auto layoutScaleAL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                auto tensorScaleAL0 =
                    asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto tensorBlockScaleAL1 = tensorScaleAL1.slice(asc::te::make_coord(0, 0),
                    asc::te::make_shape(curM, CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                auto copyL12L0MxScaleA = asc::te::make_copy(::Tile::CopyL12L0MxScaleA3510{});
                copyL12L0MxScaleA.call(tensorScaleAL0, tensorBlockScaleAL1, asc::te::make_coord(0, kL0Offset));

                auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
                    CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
                auto tensorScaleBL0 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto tensorBlockScaleBL1 = tensorScaleBL1.slice(asc::te::make_coord(0, 0),
                    asc::te::make_shape(CeilDiv(curKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                auto copyL12L0MxScaleB = asc::te::make_copy(::Tile::CopyL12L0MxScaleB3510{});
                copyL12L0MxScaleB.call(tensorScaleBL0, tensorBlockScaleBL1, asc::te::make_coord(kL0Offset, 0));
                // L1->L0 搬运完成：MTE1 释放，M 占用该 L0 做 Mmad。
                AscendC::Mutex::Unlock<PIPE_MTE1>(L0Mutex(0));
                AscendC::Mutex::Lock<PIPE_M>(L0Mutex(0));

                asc::te::unit_flag_mode mmadUnitFlag = asc::te::unit_flag_mode::disable;
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
                // Mmad 完成：M 释放该 L0，MTE1 可载下一轮。
                AscendC::Mutex::Unlock<PIPE_M>(L0Mutex(0));
            }

            // A/B/scale L1 全部 L1->L0 完成：MTE1 释放该 L1。
            AscendC::Mutex::Unlock<PIPE_MTE1>(L1Mutex(0));
        }
        // 累加完成：M 释放 L0C，FIX 占用做 L0C->GM 搬运。
        AscendC::Mutex::Unlock<PIPE_M>(L0CMutex(0));
        AscendC::Mutex::Lock<PIPE_FIX>(L0CMutex(0));

        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(copyL0C2GM, cGlobal, tensorL0C);

        // L0C->GM 完成：FIX 释放 L0C。
        AscendC::Mutex::Unlock<PIPE_FIX>(L0CMutex(0));
    }
};
}
