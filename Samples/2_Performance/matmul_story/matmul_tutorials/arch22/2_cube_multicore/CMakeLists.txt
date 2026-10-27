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

# v2 换 Cube + 多核（高级 API）：785~793 µs / ~174 TFLOPS
add_executable(step2_matmul_cube_multicore step2_matmul_cube_multicore.asc)
target_compile_options(step2_matmul_cube_multicore PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step2_matmul_cube_multicore PRIVATE m dl platform tiling_api register)
install(TARGETS step2_matmul_cube_multicore RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
