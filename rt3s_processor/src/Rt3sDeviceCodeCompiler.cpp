/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#include "Rt3sDeviceCodeCompiler.h"

#include "cmrc/cmrc.hpp"

#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <nvrtc.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

CMRC_DECLARE(BG::rt3s_processor_embedded_source);
constexpr char const* g_source_file[] {"Rt3sProcessor.cu"};
constexpr char const* g_header_files[] {
    "GpuHsTasNet.h",
    "Properties.h",
    "Rt3sProcessor.cuh",
    "ScrambledNames.h",
    "gpu_primitives/Conv1D.h",
    "gpu_primitives/Linear.h",
    "gpu_primitives/LSTM.h",
    "gpu_primitives/RMSNorm.h",
    "gpu_primitives/Spectrogram.h",
    "gpu_primitives/detail/fft_lut.h",
    "gpu_primitives/detail/fir_spec.cuh",
    "gpu_primitives/Activation.h",
    "gpu_primitives/GpuConv1x1.h",
    "gpu_primitives/GpuMatrix.h",
    "gpu_primitives/GpuMatrix.h",
    "gpu_primitives/GpuRingbuffer.h",
    "gpu_primitives/GpuVector.h",
    "gpu_primitives/GpuVector.h",
    "processor_utilities/platform/Abstraction.h",
    "processor_utilities/platform/Abstraction.h",
    "processor_utilities/platform/Abstraction.h",
    "processor_utilities/scheduler/common_macros.h",
    "processor_utilities/scheduler/common_macros.h",
    "processor_utilities/scheduler/task_command_description.h",
    //    "processor_utilities/scheduler/device/atomics.cuh",
    "processor_utilities/scheduler/device/context.cuh",
    "processor_utilities/scheduler/device/defaultcontext.cuh",
    "processor_utilities/scheduler/device/fences_and_syncs.cuh",
    "processor_utilities/scheduler/device/processor.cuh",
    "processor_utilities/scheduler/device/warp_context.cuh",
    "processor_utilities/scheduler/device/warp_context_pow2_wrapper.cuh",
    "processor_utilities/scheduler/device/warp_primitives.cuh"};
constexpr char const* g_include_names[] {
    "GpuHsTasNet.h",
    "Properties.h",
    "Rt3sProcessor.cuh",
    "ScrambledNames.h",
    "gpu_primitives/Conv1D.h",
    "gpu_primitives/Linear.h",
    "gpu_primitives/LSTM.h",
    "gpu_primitives/RMSNorm.h",
    "gpu_primitives/Spectrogram.h",
    "detail/fft_lut.h",
    "detail/fir_spec.cuh",
    "gpu_primitives/Activation.h",
    "gpu_primitives/GpuConv1x1.h",
    "gpu_primitives/GpuMatrix.h",
    "GpuMatrix.h",
    "gpu_primitives/GpuRingbuffer.h",
    "gpu_primitives/GpuVector.h",
    "GpuVector.h",
    "platform/Abstraction.h",
    "../platform/Abstraction.h",
    "../../platform/Abstraction.h",
    "scheduler/common_macros.h",
    "../common_macros.h",
    "../task_command_description.h",
    //    "atomics.cuh",
    "context.cuh",
    "defaultcontext.cuh",
    "fences_and_syncs.cuh",
    "scheduler/device/processor.cuh",
    "warp_context.cuh",
    "warp_context_pow2_wrapper.cuh",
    "warp_primitives.cuh"};

#define NVRTC_SAFE_CALL(x)                                    \
    do {                                                      \
        nvrtcResult result = x;                               \
        if (result != NVRTC_SUCCESS) {                        \
            std::cerr << "\nerror: " #x " failed with error " \
                      << nvrtcGetErrorString(result) << '\n'; \
            return GPUA::processor::v2::ErrorCode::eFail;     \
        }                                                     \
    } while (0)

namespace {

#if WIN32
static const std::string g_rt3s_code_prefix = "";
#else
static const std::string g_rt3s_code_prefix = "lib";
#endif

static const std::string g_rt3s_code_filename = g_rt3s_code_prefix + "rt3s_processor.";
static const std::string g_rt3s_code_file_ext = ".cubin";
} // namespace

Rt3sDeviceCodeCompiler::Rt3sDeviceCodeCompiler(const GPUA::processor::v2::DeviceCodeSpecification& specification) :
    m_platform {specification.platform} {
}

GPUA::processor::v2::ErrorCode Rt3sDeviceCodeCompiler::GetDeviceCode(char*& cubin, uint64_t& cubin_size) noexcept {
    static_assert(sizeof(g_source_file) && sizeof(g_source_file) / sizeof(g_source_file[0]), "Rt3sDeviceCodeCompiler::CompileDeviceCode: exactly one source file has to be provided");
    static_assert(sizeof(g_header_files) == sizeof(g_include_names), "Rt3sDeviceCodeCompiler::GetDeviceCode: each header requires exactly one include name");
    try {
        if (m_platform == L"pubkey.ed25519") {
            cubin_size = 0;
            cubin = nullptr;
            return GPUA::processor::v2::ErrorCode::eFail;
        }

        auto fs = cmrc::BG::rt3s_processor_embedded_source::get_filesystem();
        if (fs.exists(g_source_file[0]) && fs.is_file(g_source_file[0])) {
            std::string binaryname {g_rt3s_code_filename};
            using convert_type = std::codecvt_utf8<wchar_t>;
            std::wstring_convert<convert_type, wchar_t> converter;
            std::string platform_str = converter.to_bytes(m_platform);
            binaryname += platform_str; // e.g. "sm_75" or "gfx1030"
            binaryname += g_rt3s_code_file_ext;

            auto proc_src = fs.open(g_source_file[0]);

            // gather headers
            int num_headers {0};
            char const* const* headers {nullptr};
            std::vector<char const*> headers_data;
            char const* const* include_names {nullptr};
            if (sizeof(g_header_files)) {
                num_headers = sizeof(g_header_files) / sizeof(g_header_files[0]);
                headers_data.resize(num_headers);
                for (int hid {0}; hid < num_headers; ++hid) {
                    if (fs.exists(g_header_files[hid]) && fs.is_file(g_header_files[hid])) {
                        headers_data[hid] = fs.open(g_header_files[hid]).begin();
                    }
                    else {
                        cubin_size = 0;
                        cubin = nullptr;
                        return GPUA::processor::v2::ErrorCode::eFail;
                    }
                }
                headers = &headers_data[0];
                include_names = &g_include_names[0];
            }

            // Create an instance of nvrtcProgram with the SAXPY code string
            nvrtcProgram prog;
            NVRTC_SAFE_CALL(
                nvrtcCreateProgram(&prog, // prog
                    proc_src.begin(),     // buffer
                    binaryname.c_str(),   // name
                    num_headers,          // numHeaders
                    headers,              // headers
                    include_names)        // includeNames
            );

            // Gehter all compile options
            std::vector<char const*> opts;

            const char* ctime_opts[] = {GPU_AUDIO_RTC_COMPILE_FLAGS};

            opts.reserve(sizeof(ctime_opts) / sizeof(ctime_opts[0]));
            for (auto& opt : ctime_opts) {
                opts.push_back(opt);
            }

            std::string arch_opt("-arch=" + platform_str);
            opts.push_back(arch_opt.c_str());

            opts.push_back("--device-as-default-execution-space");

            //std::string cublasdx_include_dir_opt = std::string("-I") + std::getenv("CUBLASDX_INCLUDE_DIR");
            //opts.push_back(cublasdx_include_dir_opt.c_str());

            //std::string cutlass_include_dir_opt = std::string("-I") + std::getenv("CUTLASS_INCLUDE_DIR");
            //opts.push_back(cutlass_include_dir_opt.c_str());

            // Compile the program
            nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog
                opts.size(),                                      // numOptions
                opts.data());                                     // options

            // Obtain compilation log from the program
            size_t log_size {};
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
            char* log = new char[log_size];
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
#ifndef NDEBUG
            if (log_size > 1) {
                std::cout << log << '\n';
            }
#endif // !NDEBUG
            delete[] log;

            if (compileResult != NVRTC_SUCCESS) {
                cubin_size = 0;
                cubin = nullptr;
                return GPUA::processor::v2::ErrorCode::eFail;
            }
            // Obtain cubin from the program
            size_t csize {};
            NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &csize));
            m_cubin = std::make_unique<char[]>(csize);
            NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, m_cubin.get()));
            // Destroy the program.
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

            cubin_size = static_cast<uint64_t>(csize);
            cubin = m_cubin.get();
            return GPUA::processor::v2::ErrorCode::eSuccess;
        }
    }
    catch (const std::exception& exc) {
        // TODO: write to profiler
        std::cout << exc.what() << std::endl;
    }
    cubin = nullptr;
    return GPUA::processor::v2::ErrorCode::eFail;
}
