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
 * \file quant_matmul_hifp8_kc_mix_kernel.h
 * \brief MIX kernel for KC: AIC cube(A@B→L0C→UB) + AIV epilogue(UB→scale→GM), cooperative on the same core.
 */

#ifndef QUANT_MATMUL_HIFP8_KC_MIX_KERNEL_H
#define QUANT_MATMUL_HIFP8_KC_MIX_KERNEL_H

#include "kernel_operator.h"

#include "block/block_mmad.h"
#include "block/quant_matmul_hifp8_block_scheduler_swat.h"
#include "policy/dispatch_policy.h"
#include "epilogue/quant_matmul_hifp8_kc_epilogue.h"
#include "utils/constant.h"

namespace QuantMatmulHifp8Kc {

template <class ProblemShape, class BlockMmad, class BlockEpilogue>
class KcMixKernel {
public:
    __aicore__ inline KcMixKernel()
    {}
    __aicore__ inline ~KcMixKernel()
    {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using L0CType = typename BlockMmad::L0CType;
    using BlockMmadParams = typename BlockMmad::Params;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    using BlockSchedulerOp = Block::BlockSchedulerQuantHifp8Swat<ProblemShape, transA, transB>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = typename BlockMmad::LayoutA;
    using MakeLayoutB = typename BlockMmad::LayoutB;

    struct QBMMTiling {
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        EpilogueParams epiParams;
    };

    // Cross-core sync flags: AIC sets 0x8 (PIPE_FIX) after the fixpipe write to UB;
    // AIV sets 0x9 (PIPE_V) after consuming the UB; AIC waits 0x9 before the next tile.
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 0x8;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 0x9;

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIC {
            InitAic(params);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.Init(params.epiParams);
        }
        Run(params);
    }

private:
    __aicore__ inline void InitAic(const Params& params)
    {
        aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        problemShape_ = {params.problemShape.m, params.problemShape.n, params.problemShape.k};
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
        mmadOp_.Init(
            problemShape_, l0TileShape, params.qbmmParams.kAL1, params.qbmmParams.kBL1, params.qbmmParams.nBufferNum,
            enableL0CPingPong);
    }

    __aicore__ inline void Run(const Params& params)
    {
        const int64_t m = params.problemShape.m;
        const int64_t n = params.problemShape.n;
        const int64_t k = params.problemShape.k;

        BlockSchedulerOp bs(params.problemShape, params.schParams);

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto gmAFull = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmBase_), layoutA);
        auto gmBFull = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmBase_), layoutB);

        if ((bs.GetEndBlockIdx() + 1) * params.schParams.mTailTile * params.schParams.nTailTile <=
            AscendC::GetBlockNum()) {
            bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
        }

        BlockCoord blockCoord;
        bool hasBlock = false;
        while (bs.GetTileIdx(blockCoord)) {
            BlockShape singleShape = bs.GetBlockShape(blockCoord);
            if (AscendC::Std::get<MNK_M>(singleShape) <= 0 || AscendC::Std::get<MNK_N>(singleShape) <= 0) {
                break;
            }
            const int64_t mPos = AscendC::Std::get<MNK_M>(blockCoord);
            const int64_t nPos = AscendC::Std::get<MNK_N>(blockCoord);
            const int64_t curM = AscendC::Std::get<MNK_M>(singleShape);
            const int64_t curN = AscendC::Std::get<MNK_N>(singleShape);

            ProcessOneBlock(gmAFull, gmBFull, singleShape, mPos, nPos, curM, curN, k, n, hasBlock);
            hasBlock = true;
        }
        if ASCEND_IS_AIC {
            if (hasBlock) {
                // Consume the 0x9 set by AIV for the last tile, so no flag leaks into the next launch.
                AscendC::CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG);
            }
        }
    }

    template <class GmTensorA, class GmTensorB>
    __aicore__ inline void ProcessOneBlock(
        const GmTensorA& gmAFull, const GmTensorB& gmBFull, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
        int64_t curM, int64_t curN, int64_t k, int64_t n, bool hasBlock)
    {
        constexpr int64_t kPos = 0;
        // UB row pitch aligned to 32B.
        constexpr int64_t UB_FLOAT_ALIGN = 32 / sizeof(float);
        if ASCEND_IS_AIC {
            auto gmBlockA = gmAFull.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(curM, k));
            auto gmBlockB = gmBFull.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, curN));

            // ubC: destination tensor on the AIV UB (tensor_api, local offset 0).
            // M aligned to 2 (dual_dst_mode::split_m halves M), N aligned to 8 floats (32B, as blaze).
            // Each AIV actually receives curMAligned/2 rows with row pitch curNAligned
            // (the epilogue reads with the same pitch).
            const int64_t curNAligned = (curN + UB_FLOAT_ALIGN - 1) / UB_FLOAT_ALIGN * UB_FLOAT_ALIGN;
            const int64_t curMAligned = (curM + 1) / 2 * 2;
            auto layoutUbC = asc::te::make_frame_layout<asc::te::nd_layout_ptn>(curMAligned, curNAligned);
            auto ubC =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(0), layoutUbC);

            // Phase 1: Mmad accumulation (writes L0C, overlaps with the AIV epilogue of the previous tile).
            mmadOp_.Compute(gmBlockA, gmBlockB, singleShape);

            // Before phase 2: the AIV has consumed the UB of the previous tile (skipped for the
            // first tile), so fixpipe can overwrite UB; L0C reuse is guarded by the FIX_M event
            // inside the block, no wait needed here.
            if (hasBlock) {
                AscendC::CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG);
            }
            mmadOp_.CopyL0C2Ub(ubC, singleShape);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG);
            epilogueOp_(mPos, nPos, curM, curN);
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(AIV_SYNC_AIC_FLAG);
        }
    }

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    TupleShape problemShape_{};
    __gm__ AType* aGmBase_{nullptr};
    __gm__ BType* bGmBase_{nullptr};
};

} // namespace QuantMatmulHifp8Kc

#endif // QUANT_MATMUL_HIFP8_KC_MIX_KERNEL_H
