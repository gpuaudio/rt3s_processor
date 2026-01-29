/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#include "Rt3sDeviceCodeProvider.h"

#include "cmrc/cmrc.hpp"

#include <codecvt>
#include <fstream>
#include <iostream>
#include <sstream>
#include <locale>
#include <string>
#include <utility>

#if !(defined(GPU_AUDIO_NV) && defined(GPU_AUDIO_RTC))
CMRC_DECLARE(BG::rt3s_processor);
#endif

namespace {

#if WIN32
static const std::string g_rt3s_code_prefix = "";
#else
static const std::string g_rt3s_code_prefix = "lib";
#endif

static const std::string g_rt3s_code_filename = g_rt3s_code_prefix + "rt3s_processor.";
#if defined(GPU_AUDIO_NV)
static const std::string g_rt3s_code_file_ext = ".cubin";
#elif defined(GPU_AUDIO_AMD)
static const std::string g_rt3s_code_file_ext = ".o";
#elif defined(GPU_AUDIO_MAC)
static const std::string g_rt3s_code_file_ext = ".metallib";
#endif

} // namespace

Rt3sDeviceCodeProvider::Rt3sDeviceCodeProvider(const GPUA::processor::v2::DeviceCodeSpecification& specification) :
#if defined(GPU_AUDIO_NV) && defined(GPU_AUDIO_RTC)
    m_dev_code_compiler(specification),
#endif
    m_platform {specification.platform} {
}

GPUA::processor::v2::ErrorCode Rt3sDeviceCodeProvider::GetDeviceCode(GPUA::processor::v2::InputStream*& input_stream) noexcept {
#if defined(GPU_AUDIO_NV) && defined(GPU_AUDIO_RTC)
    try {
        char* cubin {};
        uint64_t cubin_size {};
        if (m_dev_code_compiler.GetDeviceCode(cubin, cubin_size) != GPUA::processor::v2::ErrorCode::eSuccess) {
            input_stream = nullptr;
            return GPUA::processor::v2::ErrorCode::eFail;
        }
        m_stream = std::make_unique<StreamAdapter>(cubin, cubin_size);
        input_stream = m_stream.get();
        return GPUA::processor::v2::ErrorCode::eSuccess;
    }
    catch (const std::exception& exc) {
        // TODO: write to profiler
        std::cout << exc.what() << std::endl;
    }
    input_stream = nullptr;
    return GPUA::processor::v2::ErrorCode::eFail;

#else
    // convert gpu platform arch from wstring to string
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::string platform_str = converter.to_bytes(m_platform);
    // assemble device code filename
    std::string filename {g_rt3s_code_filename + platform_str + g_rt3s_code_file_ext};

    try {
        // read device code to binary stream
        auto fs = cmrc::BG::rt3s_processor::get_filesystem();
        if (fs.exists(filename) && fs.is_file(filename)) {
            auto file = fs.open(filename);
            const auto size = std::distance(file.begin(), file.end());
            m_stream = std::make_unique<StreamAdapter>(file.begin(), size);
            input_stream = m_stream.get();
            return GPUA::processor::v2::ErrorCode::eSuccess;
        }
    }
    catch (const std::exception& exc) {
        std::cout << exc.what() << std::endl;
    }

    input_stream = nullptr;
    return GPUA::processor::v2::ErrorCode::eFail;
#endif
}
