/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef RT3S_RT3S_DEVICE_CODE_COMPILER_H
#define RT3S_RT3S_DEVICE_CODE_COMPILER_H

#include <processor_api/DeviceCodeSpecification.h>
#include <processor_api/Error.h>

#include <cstdint>
#include <memory>
#include <string>

class Rt3sDeviceCodeCompiler {
public:
    Rt3sDeviceCodeCompiler(const GPUA::processor::v2::DeviceCodeSpecification& specification);
    ~Rt3sDeviceCodeCompiler() = default;

    // Copy ctor and copy assignment are deleted along with move assignment operator deletion
    Rt3sDeviceCodeCompiler& operator=(Rt3sDeviceCodeCompiler&&) noexcept = delete;

    // GPUA::processor::v2::DeviceCodeProvider method
    GPUA::processor::v2::ErrorCode GetDeviceCode(char*& cubin, uint64_t& cubin_size) noexcept;

private:
    std::unique_ptr<char[]> m_cubin = nullptr;
    std::wstring m_platform;
};

#endif // RT3S_RT3S_DEVICE_CODE_COMPILER_H
