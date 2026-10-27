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

# 专题：全载模板（M=32 解码形态，stream vs fullload 单变量对照）：26.73 → 21.45 µs
add_executable(step8_matmul_full_load step8_matmul_full_load.asc)
target_compile_options(step8_matmul_full_load PRIVATE ${MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS})
target_link_libraries(step8_matmul_full_load PRIVATE m dl platform)
install(TARGETS step8_matmul_full_load RUNTIME DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX})
