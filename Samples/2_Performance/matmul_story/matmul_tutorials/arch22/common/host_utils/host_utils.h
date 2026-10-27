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
 * \file host_utils.h
 * \brief arch22 教程公共 host 工具：错误检查、fp16<->fp32 位级转换（含次正规数/±Inf/NaN/RNE）、
 *        golden 抽检与计时。各 Step 只保留自己的计算与调度差异。
 */

#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <securec.h>
#include "acl/acl.h"

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg)                                                                                  \
    do {                                                                                                       \
        if (!(cond)) {                                                                                         \
            throw std::runtime_error(                                                                          \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__)); \
        }                                                                                                      \
    } while (0)

namespace mm_tut {

// ---------------- 位级重解释辅助 ----------------
// 同尺寸栈对象间的 4 字节拷贝，memcpy_s 失败仅理论可能（指针有效且 destMax == count）；
// 防御性检查返回值，失败时返回全 0 位型（对两种转换恰好都是良性的零值）。
inline uint32_t FloatToBits(float f)
{
    uint32_t bits = 0;
    if (memcpy_s(&bits, sizeof(bits), &f, sizeof(f)) != EOK) {
        return 0;
    }
    return bits;
}

inline float BitsToFloat(uint32_t x)
{
    float f = 0.0f;
    if (memcpy_s(&f, sizeof(f), &x, sizeof(x)) != EOK) {
        return 0.0f;
    }
    return f;
}

// ---------------- fp16 <-> fp32 位级转换（IEEE 754 半精度完整实现） ----------------
// 编码：RNE 舍入；|x| >= 65520 收敛到 ±Inf；|x| < 2^-25 记 ±0；NaN 保持指数全 1 且尾数非 0。
inline uint16_t FloatToHalfBits(float f)
{
    const uint32_t x = FloatToBits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7fffffffu;
    if (a >= 0x7f800000u) {                       // NaN / ±Inf
        return static_cast<uint16_t>(sign | 0x7c00u | ((a & 0x7fffffu) != 0u ? 0x200u : 0u));
    }
    if (a >= 0x477ff800u) {                       // >= 65520（RNE 边界，取偶到 Inf）
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (a >= 0x38800000u) {                       // 半精度正规域 [2^-14, 65520)
        uint32_t h = sign | ((a - 0x38000000u) >> 13);
        if ((a & 0x1000u) != 0u && ((a & 0x0fffu) != 0u || (h & 1u) != 0u)) {
            ++h;                                  // RNE：进位至 0x7c00 的情况已被上界分支拦截
        }
        return static_cast<uint16_t>(h);
    }
    if (a < 0x33000000u) {                        // < 2^-25 记 ±0
        return static_cast<uint16_t>(sign);
    }
    const uint32_t e = a >> 23;                   // 102..112：半精度次正规域 [2^-25, 2^-14)
    const uint32_t m = (a & 0x7fffffu) | 0x800000u;
    if (e <= 102) {                               // [2^-25, 2^-24)：记 0 或最小次正规 1
        return static_cast<uint16_t>(sign | (a > 0x33000000u ? 1u : 0u));
    }
    const uint32_t shift = 126 - e;               // e ∈ [103, 112] -> shift ∈ [14, 23]，恒非零
    uint32_t h = m >> shift;
    if (((m >> (shift - 1)) & 1u) != 0u) {
        const uint32_t sticky = m & ((1u << (shift - 1)) - 1u);
        if (sticky != 0u || (h & 1u) != 0u) {
            ++h;                                  // 1023+1 = 1024 恰为最小正规 0x0400
        }
    }
    return static_cast<uint16_t>(sign | h);
}

// 解码：完整覆盖 ±0 / 次正规数 / 正规数 / ±Inf / NaN。
inline float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                          // ±0
        } else {                                  // 次正规数：规格化为 float 尾数
            uint32_t shift = 0;
            while ((mant & 0x400u) == 0) {        // mant != 0 保证循环必然终止
                mant <<= 1;
                ++shift;
            }
            mant &= 0x3ffu;
            bits = sign | ((113u - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13); // ±Inf / NaN
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return BitsToFloat(bits);
}

// ---------------- fp16 转换自检 ----------------
// 按用例类别拆为编码 / 解码 / 往返三组，任一 Step 启动时调用
// （CHECK_COND(mm_tut::Fp16SelfTest(), ...)），失败即非零退出。

inline bool Fp16SelfTestEncode()
{
    if (FloatToHalfBits(0.0f) != 0x0000) { return false; }          // +0
    if (FloatToHalfBits(-0.0f) != 0x8000) { return false; }         // -0
    if (FloatToHalfBits(1.0f) != 0x3c00) { return false; }          // 正规 1.0
    if (FloatToHalfBits(-1.0f) != 0xbc00) { return false; }
    if (FloatToHalfBits(65504.0f) != 0x7bff) { return false; }      // 最大有限值
    if (FloatToHalfBits(65520.0f) != 0x7c00) { return false; }      // RNE 边界恰收敛到 Inf
    if (FloatToHalfBits(5.9604644775390625e-08f) != 0x0001) { return false; }   // 2^-24 最小次正规
    if (FloatToHalfBits(6.097555160522461e-05f) != 0x03ff) { return false; }    // 1023·2^-24 最大次正规
    if (FloatToHalfBits(std::numeric_limits<float>::infinity()) != 0x7c00) { return false; }
    if (FloatToHalfBits(-std::numeric_limits<float>::infinity()) != 0xfc00) { return false; }
    const uint16_t nanH = FloatToHalfBits(std::numeric_limits<float>::quiet_NaN());
    if ((nanH & 0x7c00u) != 0x7c00u) { return false; }              // NaN：指数全 1
    if ((nanH & 0x3ffu) == 0u) { return false; }                    // 且尾数非 0
    return true;
}

inline bool Fp16SelfTestDecode()
{
    if (HalfBitsToFloat(0x0001) != 5.9604644775390625e-08f) { return false; }   // 最小次正规
    if (HalfBitsToFloat(0x03ff) != 6.097555160522461e-05f) { return false; }    // 最大次正规
    if (HalfBitsToFloat(0x0400) != 6.103515625e-05f) { return false; }          // 最小正规 2^-14
    if (HalfBitsToFloat(0x7bff) != 65504.0f) { return false; }                 // 最大有限值
    if (!(HalfBitsToFloat(0x7c00) == std::numeric_limits<float>::infinity())) { return false; }
    if (!(HalfBitsToFloat(0xfc00) == -std::numeric_limits<float>::infinity())) { return false; }
    if (!std::isnan(HalfBitsToFloat(0x7e00))) { return false; }                 // NaN
    const float negZero = HalfBitsToFloat(0x8000);                             // -0：值零且符号位为负
    if (negZero != 0.0f || !std::signbit(negZero)) { return false; }
    return true;
}

inline bool Fp16SelfTestRoundTrip()
{
    const float exact[] = {0.5f, -0.5f, 1.0f, -2048.0f, 65504.0f,
                           5.9604644775390625e-08f, 6.097555160522461e-05f};
    for (float f : exact) {
        if (HalfBitsToFloat(FloatToHalfBits(f)) != f) { return false; }        // 可精确表示值须无损往返
    }
    return true;
}

inline bool Fp16SelfTest()
{
    return Fp16SelfTestEncode() && Fp16SelfTestDecode() && Fp16SelfTestRoundTrip();
}

// ---------------- 输入生成与 golden 抽检 ----------------

// 生成 [-1, 1) 均匀分布 fp16 输入（固定种子 0，全程可复现）。
inline void MakeUniformInput(std::vector<uint16_t>& buf)
{
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : buf) {
        v = FloatToHalfBits(dist(gen));
    }
}

// golden 抽检：1024 个固定种子随机位置，CPU double 累加对照；
// 设备输出解码后非有限值（NaN/Inf）直接判不达标。返回不达标点数，maxErr 输出最大绝对误差。
// m/n/k 任一为 0 视为非法（同时防护 idx/n、idx%n 的除零与 m*n-1 的无符号下溢），返回 1 记为不达标。
inline uint32_t GoldenSpotCheck(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                                const std::vector<uint16_t>& out, uint32_t m, uint32_t n, uint32_t k,
                                double* maxErr)
{
    *maxErr = 0.0;
    if (m == 0 || n == 0 || k == 0) {
        return 1;
    }
    std::mt19937 posGen(1);
    std::uniform_int_distribution<uint32_t> posDist(0, m * n - 1);
    uint32_t bad = 0;
    for (int s = 0; s < 1024; ++s) {
        uint32_t idx = posDist(posGen);
        uint32_t i = idx / n, j = idx % n;
        double acc = 0.0;
        for (uint32_t p = 0; p < k; ++p) {
            acc += static_cast<double>(HalfBitsToFloat(a[static_cast<size_t>(i) * k + p])) *
                   static_cast<double>(HalfBitsToFloat(b[static_cast<size_t>(p) * n + j]));
        }
        const float got = HalfBitsToFloat(out[idx]);
        if (!std::isfinite(got)) {
            ++bad;
            continue;
        }
        double diff = std::fabs(static_cast<double>(got) - acc);
        if (diff > *maxErr) { *maxErr = diff; }
        if (diff > 1e-2 + 1e-2 * std::fabs(acc)) { ++bad; }
    }
    return bad;
}

// ---------------- 计时 ----------------

// 预热 warmup 次 → 同步 → 计时 iters 次 → 同步，输出均值 ms。
// 任一次流同步失败（异步执行错误在此时上报）返回 false，调用方不得再输出性能数据。
// warmup < 0 或 iters <= 0（iters 为均值除数）直接返回 false，防止除零。
template <typename Run>
inline bool TimedRun(Run run, int warmup, int iters, aclrtStream stream, double* ms)
{
    *ms = 0.0;
    if (warmup < 0 || iters <= 0) {
        return false;
    }
    for (int i = 0; i < warmup; ++i) {
        run();
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        run();
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    *ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    return true;
}

} // namespace mm_tut
