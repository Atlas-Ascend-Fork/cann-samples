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

# v3 蛇形窗口喂数调度：789 µs / 174.1 TFLOPS（与 v2 持平）
add_executable(step3_matmul_snake_schedule step3_matmul_snake_schedule.asc)
target_compile_options(step3_matmul_snake_schedule PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step3_matmul_snake_schedule PRIVATE m dl platform tiling_api register)
install(TARGETS step3_matmul_snake_schedule RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
