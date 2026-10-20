/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GELU_ELTWISE_REGBASE_SAMPLE_COMMON_H_
#define GELU_ELTWISE_REGBASE_SAMPLE_COMMON_H_

#include "../../common/host_compare.h"

namespace GeluEltwiseRegbaseSample {

using CannSamples::GetExeDir;
using CannSamples::ReadBin;
using CannSamples::GenerateData;
using CannSamples::CompareFloat;

constexpr uint32_t TOTAL_M = 8192;
constexpr uint32_t TOTAL_N = 8192;
constexpr uint32_t SINGLE_CORE_M = 256;
constexpr uint32_t SINGLE_CORE_N = 4096;
constexpr uint32_t TILE_LEN = 2 * SINGLE_CORE_N;
constexpr uint32_t BLOCKS =
    ((TOTAL_M + SINGLE_CORE_M - 1) / SINGLE_CORE_M) * ((TOTAL_N + SINGLE_CORE_N - 1) / SINGLE_CORE_N);
constexpr float COMPARE_TOL = 1e-3f;
constexpr int32_t MAX_ERROR_ELEM_NUM = 20;



inline int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}




template <typename T>
inline void CheckSize(const std::string& name, const std::vector<T>& data, size_t expected)
{
    if (data.size() != expected) {
        std::ostringstream oss;
        oss << name << " size mismatch, expected " << expected << ", actual " << data.size();
        throw std::runtime_error(oss.str());
    }
}

template <int kStep>
inline int RunSample(void (*launchKernel)(uint32_t, aclrtStream, uint8_t*, uint8_t*), const std::string& sampleName)
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::string exeDir = GetExeDir();
    ret = GenerateData(exeDir);
    CHECK_RET(ret == 0, aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    constexpr size_t kElems = static_cast<size_t>(TOTAL_M) * TOTAL_N;
    size_t bytes = kElems * sizeof(float);
    std::vector<float> hostIn;
    std::vector<float> golden;
    try {
        ReadBin(exeDir + "/input/input_x.bin", hostIn);
        ReadBin(exeDir + "/output/golden.bin", golden);
        CheckSize("input_x", hostIn, kElems);
        CheckSize("golden", golden, kElems);
    } catch (const std::exception& e) {
        std::cerr << "Read input/golden failed: " << e.what() << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint8_t *dIn = nullptr, *dOut = nullptr;
    ret = aclrtMalloc(reinterpret_cast<void**>(&dIn), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtMalloc(reinterpret_cast<void**>(&dOut), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    ret = aclrtMemcpy(dIn, bytes, hostIn.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtFree(dOut); aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    launchKernel(BLOCKS, stream, dIn, dOut);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtFree(dOut); aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    std::vector<float> hostOut(kElems);
    ret = aclrtMemcpy(hostOut.data(), bytes, dOut, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dIn); aclrtFree(dOut); aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);

    int errors = CompareFloat("output", hostOut.data(), golden, COMPARE_TOL);
    std::cout << "[" << sampleName << "] step " << kStep << (errors == 0 ? " PASSED" : " FAILED") << std::endl;

    aclrtFree(dIn);
    aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return errors == 0 ? 0 : 1;
}

} // namespace GeluEltwiseRegbaseSample

#endif // GELU_ELTWISE_REGBASE_SAMPLE_COMMON_H_
