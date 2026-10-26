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
 * \file leakyrelu_vec.h
 * \brief 多 AIV LeakyReLU vector kernel（双缓冲队列 + 尾块处理），split 方案的各 Step 共用
 */

#pragma once
#include "kernel_operator.h"

constexpr uint32_t LR_VEC_CHUNK = 8192;

__global__ __vector__ void leakyrelu_vec(__gm__ uint8_t* x, __gm__ uint8_t* y, uint32_t perCore, float alpha)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> qX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> qY;
    pipe.InitBuffer(qX, 2, LR_VEC_CHUNK * sizeof(float));
    pipe.InitBuffer(qY, 2, LR_VEC_CHUNK * sizeof(float));
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    uint64_t base = static_cast<uint64_t>(AscendC::GetBlockIdx()) * perCore;
    for (uint32_t off = 0; off < perCore; off += LR_VEC_CHUNK) {
        const uint32_t cnt = (perCore - off < LR_VEC_CHUNK) ? (perCore - off) : LR_VEC_CHUNK;
        AscendC::LocalTensor<float> xl = qX.AllocTensor<float>();
        AscendC::DataCopy(xl, xGm[base + off], cnt);
        qX.EnQue(xl);
        xl = qX.DeQue<float>();
        AscendC::LocalTensor<float> yl = qY.AllocTensor<float>();
        AscendC::LeakyRelu(yl, xl, alpha, cnt);
        qY.EnQue(yl);
        yl = qY.DeQue<float>();
        AscendC::DataCopy(yGm[base + off], yl, cnt);
        qX.FreeTensor(xl);
        qY.FreeTensor(yl);
    }
}
