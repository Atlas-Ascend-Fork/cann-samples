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
 * \brief MatMul+LeakyReLU 分步教程的 host 侧公共函数：fp16 位转换、随机数据填充、golden 抽检与计时
 */

#pragma once
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include "securec.h"
#include "acl/acl.h"

// ACL 关键调用检查：失败时打印并 return 1（仅用于返回 int32_t 的 host 函数，与 matmul_tutorials 的
// CHECK_COND 同为"失败即终止"策略；异步执行失败可能在同步/拷回阶段才上报，因此所有 ACL 调用都要检查）
#define CHECK_ACL(expr)                                                         \
    do {                                                                        \
        aclError ret_ = (expr);                                                 \
        if (ret_ != ACL_SUCCESS) {                                              \
            std::cerr << "[ERROR] " #expr " failed, ret=" << ret_ << " ("       \
                      << __FILE__ << ":" << __LINE__ << ")" << std::endl;       \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// 安全解析命令行正整数参数：空串、非数字、越界（> UINT32_MAX）时返回 false
inline bool ParseUint32Arg(const char* s, uint32_t& out)
{
    if (s == nullptr || *s == '\0') { return false; }
    errno = 0;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX) { return false; }
    out = static_cast<uint32_t>(v);
    return true;
}

// fp32 -> fp16 位级转换（round-toward-zero，输入范围 (-1,1) 内与 device 侧转换结果一致）
inline uint16_t FloatToHalfBits(float f)
{
    uint32_t x = 0;
    if (memcpy_s(&x, sizeof(x), &f, sizeof(f)) != EOK) { return 0; }
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 112;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) { return static_cast<uint16_t>(sign); }
    if (exp >= 31) { return static_cast<uint16_t>(sign | 0x7bff); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t x = (exp == 0) ? sign : (sign | ((exp + 112) << 23) | (mant << 13));
    float f = 0.0F;
    if (memcpy_s(&f, sizeof(f), &x, sizeof(x)) != EOK) { return 0.0F; }
    return f;
}

// 以 (-1,1) 均匀分布填充 fp16 数据；多个数组共用一个 gen 以保持数据流连续
inline void FillRandomHalf(std::vector<uint16_t>& data, std::mt19937& gen)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) { v = FloatToHalfBits(dist(gen)); }
}

// 单点 golden：C[i][j] = sum_p A[i][p] * B[p][j]，double 累加
inline double GoldenDotAt(const std::vector<uint16_t>& hostA, const std::vector<uint16_t>& hostB,
                          uint32_t i, uint32_t j, uint32_t n, uint32_t k)
{
    double acc = 0.0;
    for (uint32_t p = 0; p < k; ++p) {
        acc += static_cast<double>(HalfBitsToFloat(hostA[static_cast<uint64_t>(i) * k + p])) *
               static_cast<double>(HalfBitsToFloat(hostB[static_cast<uint64_t>(p) * n + j]));
    }
    return acc;
}

// 固定 shape 版本的 golden 抽检：均匀随机抽 1024 个点，容差 1e-2 + 1e-2 * |golden|
// applyLeaky 为 false 时 golden 只算 matmul（用于分离 EPILOGUE 的对照实验）
inline bool SpotCheck1024(const std::vector<float>& out, const std::vector<uint16_t>& hostA,
                          const std::vector<uint16_t>& hostB, uint32_t m, uint32_t n, uint32_t k,
                          float alpha, bool applyLeaky, double& maxErr, uint32_t* badCount = nullptr)
{
    if (m == 0 || n == 0) { return false; }
    std::mt19937 posGen(1);
    std::uniform_int_distribution<uint32_t> posDist(0, m * n - 1);
    uint32_t bad = 0;
    maxErr = 0.0;
    for (int s = 0; s < 1024; ++s) {
        uint32_t idx = posDist(posGen);
        uint32_t i = idx / n, j = idx % n;
        double acc = GoldenDotAt(hostA, hostB, i, j, n, k);
        double golden = applyLeaky ? (acc > 0.0 ? acc : static_cast<double>(alpha) * acc) : acc;
        double diff = std::fabs(static_cast<double>(out[idx]) - golden);
        // 非有限值（NaN/Inf）直接判失败：NaN 的比较恒为 false，不做 isfinite 检查会被静默放过
        if (!std::isfinite(out[idx]) || !std::isfinite(golden) || !std::isfinite(diff)) {
            ++bad;
            maxErr = std::numeric_limits<double>::infinity();
            continue;
        }
        if (diff > maxErr) { maxErr = diff; }
        if (diff > 1e-2 + 1e-2 * std::fabs(golden)) { ++bad; }
    }
    if (badCount != nullptr) { *badCount = bad; }
    return bad == 0;
}

// NaN/Inf 反向自检：构造必然通过的样本，再整幅注入 NaN/Inf 污染，SpotCheck1024 必须判失败；
// 防止精度检查逻辑本身退化（如漏掉 isfinite）后把非法输出报成 PASS
inline bool SpotCheckNegativeSelfTest()
{
    const uint32_t m = 16;
    const uint32_t n = 16;
    const uint32_t k = 16;
    std::vector<uint16_t> a(m * k, FloatToHalfBits(0.5f));
    std::vector<uint16_t> b(k * n, FloatToHalfBits(0.25f));
    std::vector<float> out(m * n, 0.0f);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            out[i * n + j] = static_cast<float>(GoldenDotAt(a, b, i, j, n, k));
        }
    }
    double maxErr = 0.0;
    if (!SpotCheck1024(out, a, b, m, n, k, 0.1f, true, maxErr)) { return false; } // 未污染必须通过
    std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
    if (SpotCheck1024(out, a, b, m, n, k, 0.1f, true, maxErr)) { return false; }  // NaN 必须判失败
    std::fill(out.begin(), out.end(), std::numeric_limits<float>::infinity());
    if (SpotCheck1024(out, a, b, m, n, k, 0.1f, true, maxErr)) { return false; }  // Inf 必须判失败
    return true;
}

// 计时样板：warmup 次预跑 + iters 次计时，ms 输出平均毫秒；
// 同步失败返回 false——异步执行失败可能在同步阶段才上报，此时不得输出有效性能结果
template <typename F>
inline bool TimeKernelMs(F&& run, int warmup, int iters, aclrtStream stream, double& ms)
{
    ms = 0.0;
    if (iters <= 0) { return false; }
    for (int i = 0; i < warmup; ++i) { run(); }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { return false; }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { run(); }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { return false; }
    auto t1 = std::chrono::steady_clock::now();
    // iters 在函数入口已校验，此处就地钳位除数以满足静态检查
    const int safeIters = (iters > 0) ? iters : 1;
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / safeIters;
    return true;
}
