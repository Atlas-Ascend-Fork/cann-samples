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

function(add_fusion_tutorial TARGET SRC)
    cmake_parse_arguments(ARG "" "" "LIBS" ${ARGN})
    add_executable(${TARGET} ${SRC})
    target_compile_options(${TARGET} PRIVATE ${FUSION_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${TARGET} PRIVATE ${ARG_LIBS})
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${FUSION_INSTALL_PREFIX}/fusion_tutorials)
endfunction()

add_fusion_tutorial(step01_matmul_single_core_vector
    step01_single_core_vector/step01_matmul_single_core_vector.asc
    LIBS m dl platform tiling_api)

# 高阶 API（Matmul tiling）
add_fusion_tutorial(step02_matmul_cube_fused_mix
    step02_cube_fused_mix/step02_matmul_cube_fused_mix.asc
    LIBS m dl platform tiling_api register)
add_fusion_tutorial(step03_matmul_fused_2d
    step03_fused_2d/step03_matmul_fused_2d.asc
    LIBS m dl platform tiling_api register)
add_fusion_tutorial(step04_matmul_decoupled_mix
    step04_decoupled_mix/step04_matmul_decoupled_mix.asc
    LIBS m dl platform tiling_api register)
add_fusion_tutorial(step05_matmul_snake_mix
    step05_snake_mix/step05_matmul_snake_mix.asc
    LIBS m dl platform tiling_api register)

# 低阶 API（Nd2Nz / LoadData / Mmad / Fixpipe 手工流水）
add_fusion_tutorial(step06_matmul_decoupled_lowlevel_aswt
    step06_decoupled_lowlevel/step06_matmul_decoupled_lowlevel_aswt.asc
    LIBS m dl platform)
add_fusion_tutorial(step07_matmul_leakyrelu_fused_lowlevel
    step07_fused_lowlevel/step07_matmul_leakyrelu_fused_lowlevel.asc
    LIBS m dl platform)
add_fusion_tutorial(step08_matmul_leakyrelu_fused_lowlevel_aswt
    step08_fused_lowlevel_aswt/step08_matmul_leakyrelu_fused_lowlevel_aswt.asc
    LIBS m dl platform)

add_fusion_tutorial(step09_matmul_leakyrelu_fused_general
    step09_general/step09_matmul_leakyrelu_fused_general.asc
    LIBS m dl platform)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    DESTINATION ${FUSION_INSTALL_PREFIX}/fusion_tutorials
)
