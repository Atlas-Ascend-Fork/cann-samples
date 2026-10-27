# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cann_sample_check_arch(dav-2201)

set(MATMUL_TUTORIAL_INSTALL_PREFIX "${MATMUL_INSTALL_PREFIX}/matmul_tutorials/arch22")

set(MATMUL_TUTORIAL_ASCENDC_COMPILE_OPTS
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
)

# 迭代阶梯（每步一个子目录，实测数据见 README.md）：
#   v1 单核向量 → v2 Cube 多核 → v3 蛇形窗口 → v4 错位分核单开（高级 API，1~4）
#   v5 低阶流水 → v6 +4buf+错位分核（最优）→ v7 任意 shape 泛化（低阶，5~7）；
#   8_full_load = 全载模板专题（M=32 解码形态，stream/fullload 单变量对照）
# 编译选项不含 -O3：与本目录 README 实测数据口径一致（bisheng 直编命令同）。
# 链接差异：step1~4 用 MultiCoreMatmulTiling（tiling_api），其中 step2~4 的高级
# matmul API 另需 register；step5~8 纯低阶通路只需 platform。

add_subdirectory(1_single_core_vector)
add_subdirectory(2_cube_multicore)
add_subdirectory(3_snake_schedule)
add_subdirectory(4_stagger_schedule)
add_subdirectory(5_lowlevel_pipeline)
add_subdirectory(6_lowlevel_4buf_stagger)
add_subdirectory(7_general_shape)
add_subdirectory(8_full_load)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION ${MATMUL_TUTORIAL_INSTALL_PREFIX}/scripts
    FILES_MATCHING
    PATTERN "*.sh"
)
