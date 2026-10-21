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

# Scripts live beside the executables, both in the build tree and after install.
set(SOFTMAX_INSTALL_DIR 2_Performance/softmax_membase_story)
foreach(SCRIPT_NAME gen verify)
    configure_file(scripts/${SCRIPT_NAME}.py softmax_${SCRIPT_NAME}.py COPYONLY)
    install(FILES scripts/${SCRIPT_NAME}.py
        DESTINATION ${SOFTMAX_INSTALL_DIR} RENAME softmax_${SCRIPT_NAME}.py)
endforeach()

# v00-v06 form the tutorial; v04_1 demonstrates manual UB and event management.
set(SOFTMAX_VERSIONS v00 v01 v02 v03 v04 v04_1 v05 v06)
set(SOFTMAX_VERSION_TARGETS "")
file(GLOB VERSION_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true
    ${CMAKE_CURRENT_SOURCE_DIR}/src/v*)
foreach(VERSION_DIR IN LISTS VERSION_DIRS)
    if(IS_DIRECTORY ${VERSION_DIR})
        get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
        if(NOT VERSION_NAME IN_LIST SOFTMAX_VERSIONS)
            message(FATAL_ERROR "Unregistered Softmax version: src/${VERSION_NAME}")
        endif()
    endif()
endforeach()

foreach(VERSION_NAME IN LISTS SOFTMAX_VERSIONS)
    set(TARGET_NAME softmax_membase_${VERSION_NAME})
    file(GLOB HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${VERSION_NAME}/*.h)
    add_executable(${TARGET_NAME} src/${VERSION_NAME}/softmax.asc ${HEADER_FILES})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
    )
    if(SIM)
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:ASC>:-g>")
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE
        m dl platform tiling_api ascendcl stdc++
    )
    if(NOT SIM)
        target_link_libraries(${TARGET_NAME} PRIVATE runtime)
    endif()
    if(VERSION_NAME MATCHES "^v(03|04|04_1|05|06)$")
        # Reduce API workspace queries use ge::Shape.
        target_link_libraries(${TARGET_NAME} PRIVATE graph_base)
    endif()
    list(APPEND SOFTMAX_VERSION_TARGETS ${TARGET_NAME})
endforeach()

add_custom_target(softmax_membase_story DEPENDS ${SOFTMAX_VERSION_TARGETS})
install(TARGETS ${SOFTMAX_VERSION_TARGETS} RUNTIME DESTINATION ${SOFTMAX_INSTALL_DIR})
install(PROGRAMS scripts/quicksmoke_softmax.sh DESTINATION ${SOFTMAX_INSTALL_DIR})
