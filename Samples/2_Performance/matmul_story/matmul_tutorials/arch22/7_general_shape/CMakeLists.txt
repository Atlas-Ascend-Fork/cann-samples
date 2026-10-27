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

# v7 = v6 任意 shape 泛化（M/N/K 为 16 倍数）：4096³ 回归 616.6 µs / 222.9 TFLOPS
add_executable(step7_matmul_general_shape step7_matmul_general_shape.asc)
target_compile_options(step7_matmul_general_shape PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step7_matmul_general_shape PRIVATE m dl platform)
install(TARGETS step7_matmul_general_shape RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
