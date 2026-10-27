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

# v6 = v5 + L1 四级缓冲 + 错位分核（最优）：610~637 µs / 215.8~225.4 TFLOPS
add_executable(step6_matmul_lowlevel_4buf_stagger step6_matmul_lowlevel_4buf_stagger.asc)
target_compile_options(step6_matmul_lowlevel_4buf_stagger PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step6_matmul_lowlevel_4buf_stagger PRIVATE m dl platform)
install(TARGETS step6_matmul_lowlevel_4buf_stagger RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
