/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mutex_id.h
 * \brief 共享的 ISASI Mutex ID 分段 helper。
 *
 * a_full_load 与 swat 两个 block 头会被 a_fullload 样例聚合进同一编译单元，若各自在
 * 文件作用域重复定义这些 helper 会导致重定义错误，故统一抽取到此处（#pragma once 去重）。
 *
 * Mutex ID 分段约定：
 *   L1  (MTE2<->MTE1): [0, 8)
 *   L0  (MTE1<->M):    [8, 16)
 */

#pragma once

namespace Block {
__aicore__ inline uint8_t L1Mutex(uint16_t id) { return static_cast<uint8_t>(id); }
__aicore__ inline uint8_t L0Mutex(uint16_t id) { return static_cast<uint8_t>(id + 8); }
} // namespace Block
