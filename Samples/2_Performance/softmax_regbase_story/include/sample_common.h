/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFTMAX_REGBASE_SAMPLE_COMMON_H_
#define SOFTMAX_REGBASE_SAMPLE_COMMON_H_

#include "../../common/host_compare.h"

namespace SoftmaxRegbaseSample {

using CannSamples::GetExeDir;
using CannSamples::ReadBin;
using CannSamples::GenerateData;
using CannSamples::CompareFloat;

constexpr uint32_t TOTAL_M = 256;
constexpr uint32_t TOTAL_N = 2048;
constexpr uint32_t SINGLE_CORE_M = 8;
constexpr uint32_t SINGLE_CORE_N = 2048;
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
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}




struct SampleRuntime {
    int32_t deviceId;
    aclrtStream stream = nullptr;
    uint8_t* dIn = nullptr;
    uint8_t* dOut = nullptr;
    bool streamSynchronized = false;
    bool streamSyncFailed = false;

    void Cleanup()
    {
        if (stream != nullptr && !streamSynchronized && !streamSyncFailed) {
            streamSynchronized = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
            streamSyncFailed = !streamSynchronized;
        }
        if (stream != nullptr && streamSyncFailed) {
            // A failed stream sync leaves the stream unsafe to destroy directly.
            aclrtResetDevice(deviceId);
            stream = nullptr;
            dIn = nullptr;
            dOut = nullptr;
            aclFinalize();
            return;
        }
        if (dIn != nullptr) {
            aclrtFree(dIn);
            dIn = nullptr;
        }
        if (dOut != nullptr) {
            aclrtFree(dOut);
            dOut = nullptr;
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

inline int ReadSampleData(const std::string& exeDir, std::vector<float>& hostIn, std::vector<float>& golden)
{
    try {
        ReadBin(exeDir + "/input/input_x.bin", hostIn);
        ReadBin(exeDir + "/output/golden.bin", golden);
    } catch (const std::exception& e) {
        std::cerr << "Read input/golden failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

inline int AllocateSampleBuffers(SampleRuntime& runtime, size_t bytes, const std::vector<float>& hostIn)
{
    auto ret = aclrtMalloc(reinterpret_cast<void**>(&runtime.dIn), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(reinterpret_cast<void**>(&runtime.dOut), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return aclrtMemcpy(runtime.dIn, bytes, hostIn.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

inline int LaunchSample(void (*launchKernel)(uint32_t, aclrtStream, uint8_t*, uint8_t*), SampleRuntime& runtime,
    uint32_t blocks, size_t bytes, std::vector<float>& hostOut)
{
    launchKernel(blocks, runtime.stream, runtime.dIn, runtime.dOut);
    auto ret = aclrtSynchronizeStream(runtime.stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        runtime.streamSyncFailed = true;
        return ret;
    }
    runtime.streamSynchronized = true;
    return aclrtMemcpy(hostOut.data(), bytes, runtime.dOut, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

inline int ExecuteSample(int step, void (*launchKernel)(uint32_t, aclrtStream, uint8_t*, uint8_t*),
    const std::string& sampleName, SampleRuntime& runtime)
{
    const auto exeDir = GetExeDir();
    auto ret = GenerateData(exeDir);
    if (ret != 0) {
        return ret;
    }
    std::vector<float> hostIn;
    std::vector<float> golden;
    ret = ReadSampleData(exeDir, hostIn, golden);
    if (ret != 0) {
        return ret;
    }
    const size_t bytes = TOTAL_M * TOTAL_N * sizeof(float);
    ret = AllocateSampleBuffers(runtime, bytes, hostIn);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<float> hostOut(TOTAL_M * TOTAL_N);
    ret = LaunchSample(launchKernel, runtime, BLOCKS, bytes, hostOut);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    int errors = CompareFloat("output", hostOut.data(), golden, COMPARE_TOL);
    std::cout << "[" << sampleName << "] step " << step << (errors == 0 ? " PASSED" : " FAILED") << std::endl;
    return errors == 0 ? ACL_SUCCESS : 1;
}

template <int kStep>
inline int RunSample(void (*launchKernel)(uint32_t, aclrtStream, uint8_t*, uint8_t*), const std::string& sampleName)
{
    const char* devEnv = std::getenv("SAMPLE_DEVICE_ID");
    SampleRuntime runtime{devEnv ? std::atoi(devEnv) : 7};
    auto ret = InitAcl(runtime.deviceId, &runtime.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    ret = ExecuteSample(kStep, launchKernel, sampleName, runtime);
    runtime.Cleanup();
    return ret;
}

} // namespace SoftmaxRegbaseSample

#endif // SOFTMAX_REGBASE_SAMPLE_COMMON_H_
