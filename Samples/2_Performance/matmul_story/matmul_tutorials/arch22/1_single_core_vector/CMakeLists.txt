# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# v1 基线：单核向量（9922.13 ms / 0.014 TFLOPS，910B2 实测）
add_executable(step1_matmul_single_core_vector step1_matmul_single_core_vector.asc)
target_compile_options(step1_matmul_single_core_vector PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step1_matmul_single_core_vector PRIVATE m dl platform tiling_api)
install(TARGETS step1_matmul_single_core_vector RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
