/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_SAMPLES_HOST_COMPARE_H_
#define CANN_SAMPLES_HOST_COMPARE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"

#ifndef SOURCE_DIR
#define SOURCE_DIR "."
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace CannSamples {

inline std::string GetExeDir()
{
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        return std::string(dirname(path));
    }
    return ".";
}

template <typename T>
inline void ReadBin(const std::string& filename, std::vector<T>& data)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Can not open file: " + filename);
    }
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t elemNum = static_cast<size_t>(fileSize) / sizeof(T);
    data.resize(elemNum);
    if (elemNum > 0) {
        file.read(reinterpret_cast<char*>(data.data()), elemNum * sizeof(T));
    }
}

inline std::string FindGenDataScript(const std::vector<std::string>& candidates)
{
    for (const auto& path : candidates) {
        std::ifstream script(path);
        if (script.is_open()) {
            return path;
        }
    }
    return candidates.back();
}

inline int RunGenerator(const std::string& script, const std::string& output, const std::string& arguments)
{
    std::ostringstream cmd;
    cmd << "env -u LD_LIBRARY_PATH python3 " << script << arguments << " --output " << output;
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        std::cerr << "Generate data failed, command: " << cmd.str() << std::endl;
    }
    return ret;
}

inline int GenerateData(const std::string& exeDir, const std::string& arguments = "", bool scriptsOnly = false)
{
    std::vector<std::string> candidates = scriptsOnly
        ? std::vector<std::string>{exeDir + "/scripts/gen_data.py", exeDir + "/../scripts/gen_data.py"}
        : std::vector<std::string>{exeDir + "/gen_data.py", exeDir + "/scripts/gen_data.py"};
    candidates.push_back(std::string(SOURCE_DIR) + "/scripts/gen_data.py");
    return RunGenerator(FindGenDataScript(candidates), exeDir, arguments);
}

inline int CompareFloat(
    const std::string& name, const float* actual, const std::vector<float>& golden, float tol, int maxErrorElements = 20)
{
    int errorCount = 0;
    float maxDiff = 0.0f;
    size_t maxDiffIndex = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        float diff = std::abs(actual[i] - golden[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIndex = i;
        }
        if (!std::isfinite(diff) || diff > tol) {
            if (errorCount < maxErrorElements) {
                std::cout << name << " mismatch index " << i << ", expected " << golden[i] << ", actual "
                          << actual[i] << ", diff " << diff << std::endl;
            }
            ++errorCount;
        }
    }
    float precision = golden.empty() ? 100.0f
                                     : static_cast<float>(golden.size() - errorCount) / golden.size() * 100.0f;
    std::cout << name << " precision " << precision << "%, errors " << errorCount << ", max diff " << maxDiff
              << " at " << maxDiffIndex << std::endl;
    return errorCount;
}

} // namespace CannSamples

#endif // CANN_SAMPLES_HOST_COMPARE_H_
