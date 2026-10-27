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

# v5 低阶 API 逐 tile 流水：653 µs / ~210 TFLOPS
add_executable(step5_matmul_lowlevel_pipeline step5_matmul_lowlevel_pipeline.asc)
target_compile_options(step5_matmul_lowlevel_pipeline PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step5_matmul_lowlevel_pipeline PRIVATE m dl platform)
install(TARGETS step5_matmul_lowlevel_pipeline RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
