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
 * \file quant_matmul_mx_kernel_base_impl.h
 * \brief TensorAPI version of the MXFP4 quantized matmul kernel.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../block/block_scheduler_mx_base.h"
#include "../block/block_mmad_mx_base.h"
#include "../utils/quant_matmul_constant.h"

namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMatmulMxKernelBaseImpl {
public:
    __aicore__ inline QuantMatmulMxKernelBaseImpl()
    {}
    __aicore__ inline ~QuantMatmulMxKernelBaseImpl()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr int32_t SCALE_C0 = 2;
    using BlockCoord = AscendC::Coord<int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>;
    using MakeLayoutB = asc::te::frame_layout_format<asc::te::dn_ext_layout_ptn>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ fp8_e8m0_t* scaleAGmAddr_;
    __gm__ fp8_e8m0_t* scaleBGmAddr_;
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelBaseImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params);
    Process(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelBaseImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelBaseImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleA = MakeLayoutScaleA{}(
        params.problemShape.m, CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
    auto layoutScaleB = MakeLayoutScaleB{}(
        CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, params.problemShape.n);
    auto layoutC = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(params.problemShape.m, params.problemShape.n);

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (asc::te::get<MNK_M>(singleShape) <= 0 || asc::te::get<MNK_N>(singleShape) <= 0) {
            return;
        }

        int64_t mTileIdx = asc::te::get<MNK_M>(blockIdx);
        int64_t nTileIdx = asc::te::get<MNK_N>(blockIdx);
        int64_t mSplitOffset = asc::te::get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape);
        int64_t nSplitOffset = asc::te::get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape);
        int64_t mPos = mTileIdx * params.qbmmParams.baseM + mSplitOffset;
        int64_t nPos = nTileIdx * params.qbmmParams.baseN + nSplitOffset;
        auto curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        auto curN = asc::te::get<IDX_N_TILEIDX>(singleShape);

        auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos),
            asc::te::make_shape(curM, params.problemShape.k));
        auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos),
            asc::te::make_shape(
                curM, CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
        auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos),
            asc::te::make_shape(params.problemShape.k, curN));
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos),
            asc::te::make_shape(
                CeilDiv(params.problemShape.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
        auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos),
            asc::te::make_shape(curM, curN));

        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
    }
}

} // namespace Kernel

