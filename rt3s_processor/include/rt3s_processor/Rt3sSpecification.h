/*
 * Copyright (c) 2024 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef RT3S_RT3S_SPECIFICATION_H
#define RT3S_RT3S_SPECIFICATION_H

#include <cstdint>

namespace Rt3sConfig {
struct Specification {
    static constexpr uint32_t Magic = 0xC5EE71A2;
    uint32_t ThisMagic {Magic};

    uint64_t params_bytes {};
    char const* params {};
};
} // namespace Rt3sConfig

#endif // RT3S_RT3S_SPECIFICATION_H
