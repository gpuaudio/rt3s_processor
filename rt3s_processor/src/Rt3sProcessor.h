/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef RT3S_RT3S_PROCESSOR_H
#define RT3S_RT3S_PROCESSOR_H

#include "Rt3sInputPort.h"
#include "Properties.h"

#include <rt3s_processor/Rt3sSpecification.h>
#include <processor_api/GpuTaskData.h>
#include <processor_api/MemoryManager.h>
#include <processor_api/ModuleBase.h>
#include <processor_api/LaunchData.h>
#include <processor_api/Processor.h>
#include <processor_api/ProcessorBlueprint.h>
#include <processor_api/ProcessorProfiler.h>
#include <processor_api/PortFactory.h>

#include <cstdint>
#include <fstream>

class Rt3sProcessor : public GPUA::processor::v2::Processor {
public:
    explicit Rt3sProcessor(GPUA::processor::v2::ProcessorSpecification& specification, GPUA::processor::v2::Module& module);
    ~Rt3sProcessor() = default;

    // Copy ctor and copy assignment are deleted along with move assignment operator deletion
    Rt3sProcessor& operator=(Rt3sProcessor&&) = delete;

    ////////////////////////////////
    // GPUA::processor::v2::Processor methods
    GPUA::processor::v2::Module& GetModule() const noexcept override;

    GPUA::processor::v2::ErrorCode SetData(void* data, uint32_t data_size) noexcept override;
    GPUA::processor::v2::ErrorCode GetData(void* data, uint32_t& data_size) const noexcept override;

    uint32_t GetInputPortCount() const noexcept override;
    GPUA::processor::v2::ErrorCode GetInputPort(uint32_t index, GPUA::processor::v2::InputPort*& port) noexcept override;

    GPUA::processor::v2::ErrorCode OnBlueprintRebuild(const GPUA::processor::v2::ProcessorBlueprint*& blueprint) noexcept override;
    GPUA::processor::v2::ErrorCode PrepareForProcess(const GPUA::processor::v2::LaunchData& data, uint32_t expected_chunks) noexcept override;
    GPUA::processor::v2::ErrorCode PrepareChunk(void* proc_data, void** task_data, uint32_t chunk_id) noexcept override;
    void OnProcessingEnd(bool after_fat_transfer) noexcept override;

    GPUA::processor::v2::ProcessorProfiler* GetProcessorProfiler() noexcept override;
    // GPUA::processor::v2::Processor methods
    ////////////////////////////////

private:
    class ParamStreamBuf : public std::streambuf {
    public:
        ParamStreamBuf(char const* data, std::size_t size) {
            char* buffer = const_cast<char*>(data);
            this->setg(buffer, buffer, buffer + size);
        }
    };

    void AdvanceCounters();

    /**
     * @brief Assign ranges of continuous device memory to task buffers
     * @param d_device_buffers [in] pointer to continuous device memory
     */
    void SetTaskBuffers(uint64_t d_device_buffers);

    /**
     * @brief Initialize common data
     * @param spec [in] processor specification
     */
    void SetUpCommon(Rt3sConfig::Specification const* spec);

    GPUA::processor::v2::Module& m_module;
    GPUA::processor::v2::MemoryManager& m_memory_manager;
    GPUA::processor::v2::PortFactory& m_port_factory;

    GPUA::processor::v2::GpuTaskData m_gpu_task[25] {};
    GPUA::processor::v2::ProcessorBlueprint m_proc_data;

    std::unique_ptr<Rt3sInputPort> m_input_port;
    GPUA::processor::v2::OutputPortPointer m_output_port {0, 0};

    bool m_changed {true};

    GPUA::processor::v2::MemoryManager::GpuMemoryPointer m_d_twiddle_LUT {0, 0};
    GPUA::processor::v2::MemoryManager::GpuMemoryPointer m_d_twiddle_inverse_LUT {0, 0};
    GPUA::processor::v2::MemoryManager::GpuMemoryPointer m_d_hstasnet {0, 0};

    uint32_t m_ringbuffer_cursor {0u};
    uint32_t m_running_counter {0u};

    // Task Parameters
    // pre fusion spec
    gpua::SpecParams m_spec_params;
    gpua::SpecEncParams m_spec_enc_params;

    // pre fusion waveform
    gpua::ConvEncodeParams m_conv_encode_params;
    gpua::ConvActivationParams m_conv_activation_params;
    gpua::BasisEmbedParams m_basis_embed_params;

    // pre-fusion lstm
    gpua::PreSpecWaveformLstmParams m_pre_spec_waveform_lstm_params;

    // fusion
    gpua::FusionParams m_fusion_params;

    // post-fusion lstm
    gpua::PostSpecWaveformLstmParams m_post_spec_waveform_lstm_params;

    // post-post-fusion rms norm
    gpua::SpecWaveformNormParams m_spec_waveform_norm_params;

    // post fusion spec
    gpua::SpecMaskParams m_spec_mask_params;
    gpua::SpecMaskApplyParams m_spec_mask_apply_params;
    gpua::InvSpecParams m_inv_spec_params;

    // post fusion waveform
    gpua::WaveformMaskParams m_waveform_mask_params;
    gpua::WaveformMaskApplyParams m_waveform_mask_apply_params;
    gpua::ConvDecodeParams m_conv_decode_params;

    // combine branches
    gpua::CombineBranchesParams m_combine_branches_params;

    // DEBUG BUFFER
    std::vector<float> h_debug_buffer;

    // device buffers
    GPUA::processor::v2::MemoryManager::GpuMemoryPointer d_device_buffer0 {0, 0};
    GPUA::processor::v2::MemoryManager::GpuMemoryPointer d_device_buffer1 {0, 0};
    std::array<GPUA::processor::v2::GpuMemory*, 2u> d_device_buffer;
};

#endif // RT3S_RT3S_PROCESSOR_H
